"""Phase 5 pipeline: train + evaluate the Flood Response Function.

Trains the FRF on synthetic storms (so the temporal encoder learns timing), then
validates on the three real historical storms — held out entirely from training.
Reports depth RMSE and onset/clearance timing error (hours) plus an event-level F1
comparable to the Phase-4 baseline, saves the model, and prints a sample
depth-vs-time curve. Run with ``python -m aquaroute.model.frf_pipeline``
(or ``make frf``).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from aquaroute.config import get_config
from aquaroute.model.frf_train import (
    HORIZON,
    NODE_FEATURES,
    evaluate_frf,
    prepare_data,
    train_frf,
)


def save_frf(model, stats, path, gnn, temporal, hidden):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": stats[0], "std": stats[1],
        "node_features": NODE_FEATURES, "horizon": HORIZON,
        "hidden": hidden, "gnn": gnn, "temporal": temporal,
    }, path)


def run(epochs: int = 50, storms_per_epoch: int = 3) -> dict:
    cfg = get_config()
    hidden = int(cfg.get("model", "frf", "hidden_dim", default=64))
    gnn = cfg.get("model", "frf", "gnn", default="graphsage")
    temporal = cfg.get("model", "frf", "temporal_encoder", default="lstm")
    print("AquaRoute Phase 5 — Flood Response Function "
          f"(temporal={temporal}, gnn={gnn}, hidden={hidden})\n")

    print("[*] Preparing data (features, segment graph, reservoir params, real storms)...")
    data = prepare_data()
    events = list(data["events"])
    print(f"    nodes: {len(data['ids']):,}  edges: {data['edge_index'].shape[1]:,}  "
          f"held-out real events: {events}\n")

    print(f"[*] Training FRF on synthetic storms ({epochs} epochs × {storms_per_epoch})...")
    model, stats = train_frf(data, epochs=epochs, storms_per_epoch=storms_per_epoch,
                             hidden=hidden, gnn=gnn, temporal=temporal)

    print("\n[*] Validating on held-out historical storms:")
    per_event = [evaluate_frf(model, data, ev, stats) for ev in events]
    for m in per_event:
        print(f"    {m['event']:<26} depth RMSE={m['depth_rmse_m']} m | "
              f"onset MAE={m['onset_mae_h']} h | clearance MAE={m['clearance_mae_h']} h | "
              f"event F1={m['event_f1']} | AUC={m['event_roc_auc']}")

    mean = {
        "depth_rmse_m": round(float(np.mean([m["depth_rmse_m"] for m in per_event])), 4),
        "onset_mae_h": _mean([m["onset_mae_h"] for m in per_event]),
        "clearance_mae_h": _mean([m["clearance_mae_h"] for m in per_event]),
        "event_f1": round(float(np.mean([m["event_f1"] for m in per_event])), 4),
        "event_roc_auc": round(float(np.mean([m["event_roc_auc"] for m in per_event
                                              if m["event_roc_auc"] is not None])), 4),
    }
    print(f"\n    MEAN  depth RMSE={mean['depth_rmse_m']} m | onset MAE={mean['onset_mae_h']} h "
          f"| clearance MAE={mean['clearance_mae_h']} h | event F1={mean['event_f1']} "
          f"| AUC={mean['event_roc_auc']}")

    base_f1 = _baseline_f1()
    if base_f1 is not None:
        print(f"    baseline XGBoost LOEO F1={base_f1} (no timing). FRF F1={mean['event_f1']} "
              f"AND adds onset/clearance timing.")

    model_path = cfg.cache_dir / "models" / "frf.pt"
    save_frf(model, stats, model_path, gnn, temporal, hidden)
    metrics_path = cfg.cache_dir / "models" / "frf_metrics.json"
    metrics_path.write_text(json.dumps({"per_event": per_event, "mean": mean,
                                        "baseline_f1": base_f1}, indent=2))
    print(f"\n    model:   {model_path}\n    metrics: {metrics_path}")

    _print_sample_curve(model, data, stats, events[-1])
    print("\nPhase 5 OK. The FRF outputs a depth-vs-time curve per segment with "
          "onset/peak/clearance. Phase 6 wires this to /predict + /segment/{id}/curve.")
    return {"per_event": per_event, "mean": mean}


def _mean(vals):
    v = [x for x in vals if x is not None]
    return round(float(np.mean(v)), 3) if v else None


def _baseline_f1():
    try:
        p = get_config().cache_dir / "models" / "baseline_metrics.json"
        return json.loads(p.read_text())["results"]["xgboost"]["loeo"]["mean"]["f1"]
    except Exception:
        return None


def _print_sample_curve(model, data, stats, event):
    from aquaroute.model.frf import derive_events, predict_all, predict_curve
    from aquaroute.model.frf_train import _standardize

    x_np, _ = _standardize(data["node_x_raw"], stats)
    pred = predict_all(model, x_np, data["edge_index"], data["events"][event]["hyeto"])
    peaks = pred.max(axis=1)
    i = int(np.argmax(peaks))
    sid = data["feats"].iloc[i]["segment_id"]
    curve = predict_curve(pred[i])
    ev = derive_events(pred[i])
    print(f"\n[sample] segment {sid} — {event}")
    print(f"    onset={ev['onset']}h  peak={ev['peak']}h (depth {ev['peak_depth']:.2f} m)  "
          f"clearance={ev['clearance']}h")
    pk = max(ev["peak_depth"], 1e-6)
    spark = "".join(" .:-=+*#"[min(7, int(d / pk * 7))] for d in curve["depth"])
    print(f"    depth curve: [{spark}]  (0..{pk:.2f} m)")


if __name__ == "__main__":
    run()
    sys.exit(0)
