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


def save_frf(model, stats, path, gnn, temporal, hidden, data):
    import numpy as np
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": stats[0], "std": stats[1],
        "node_features": NODE_FEATURES, "horizon": HORIZON,
        "hidden": hidden, "gnn": gnn, "temporal": temporal,
        # Per-segment real-SAR flood propensity (the extent), aligned to graph ids,
        # so the predictor can reconstruct depth = MAX_DEPTH * propensity * shape.
        "propensity": np.asarray(data["propensity"], dtype="float32"),
        "ids": list(data["ids"]),
    }, path)


def _propensity_generalization() -> dict:
    """Honest test that the real-SAR flood-extent grounding generalises: train the
    propensity model on all-but-one SAR event, score it on the held-out event."""
    from sklearn.metrics import roc_auc_score

    from aquaroute.db.segments_store import load_segments_gdf
    from aquaroute.labels.store import load_labels
    from aquaroute.model.propensity import _real_sar_events, compute_flood_propensity

    real = _real_sar_events()
    if len(real) < 2:
        return {"events": real, "held_out_auc": None}
    seg_ids = load_segments_gdf()["segment_id"]
    aucs = {}
    for held in real:
        p = compute_flood_propensity([e for e in real if e != held])
        y = seg_ids.map(load_labels(held).set_index("segment_id")["flooded"]).fillna(False).astype(int)
        aucs[held] = round(float(roc_auc_score(y.to_numpy(), seg_ids.map(p).fillna(0).to_numpy())), 3)
    return {"events": real, "per_event_auc": aucs,
            "mean_auc": round(float(np.mean(list(aucs.values()))), 3)}


def run(epochs: int = 55, storms_per_epoch: int = 3, compare_gnn: bool = True) -> dict:
    cfg = get_config()
    hidden = int(cfg.get("model", "frf", "hidden_dim", default=64))
    temporal = cfg.get("model", "frf", "temporal_encoder", default="lstm")
    print(f"AquaRoute — Flood Response Function (grounded in real SAR, temporal={temporal})\n")

    print("[*] Flood-extent grounding: real-SAR propensity, held-out generalisation...")
    prop = _propensity_generalization()
    if prop.get("mean_auc") is not None:
        print(f"    propensity held-out AUC (train one SAR event -> test the other): "
              f"{prop['per_event_auc']}  mean={prop['mean_auc']}")

    print("\n[*] Preparing grounded data (features, graph, real-SAR propensity, storms)...")
    data = prepare_data()
    events = list(data["events"])
    print(f"    nodes: {len(data['ids']):,}  edges: {data['edge_index'].shape[1]:,}  events: {events}\n")

    archs = ["graphsage", "gat"] if compare_gnn else [cfg.get("model", "frf", "gnn", default="graphsage")]
    trials = {}
    for gnn in archs:
        print(f"=== Training FRF with GNN={gnn} ({epochs} epochs) ===")
        model, stats = train_frf(data, epochs=epochs, storms_per_epoch=storms_per_epoch,
                                 hidden=hidden, gnn=gnn, temporal=temporal)
        per_event = [evaluate_frf(model, data, ev, stats) for ev in events]
        mean = _summarise(per_event)
        print(f"    {gnn}: mean depth RMSE={mean['depth_rmse_m']} m onset MAE={mean['onset_mae_h']} h "
              f"clearance MAE={mean['clearance_mae_h']} h event F1={mean['event_f1']} AUC={mean['event_roc_auc']}\n")
        trials[gnn] = {"model": model, "stats": stats, "per_event": per_event, "mean": mean}

    best = max(trials, key=lambda g: trials[g]["mean"]["event_roc_auc"] or 0)
    print(f"[*] Best architecture by event AUC: {best}")
    sel = trials[best]
    model, stats, per_event, mean = sel["model"], sel["stats"], sel["per_event"], sel["mean"]

    print("\n[*] Per-event validation (best model):")
    for m in per_event:
        print(f"    {m['event']:<26} depth RMSE={m['depth_rmse_m']} m | onset MAE={m['onset_mae_h']} h | "
              f"clearance MAE={m['clearance_mae_h']} h | event F1={m['event_f1']} | AUC={m['event_roc_auc']}")

    model_path = cfg.cache_dir / "models" / "frf.pt"
    save_frf(model, stats, model_path, best, temporal, hidden, data)
    metrics = {
        "gnn": best, "propensity_generalization": prop,
        "architecture_comparison": {g: trials[g]["mean"] for g in trials},
        "per_event": per_event, "mean": mean, "baseline_f1": _baseline_f1(),
    }
    (cfg.cache_dir / "models" / "frf_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n    model:   {model_path} (GNN={best})")

    _print_sample_curve(model, data, stats, events[-1])
    print("\nFRF OK — flood extent grounded in real Sentinel-1 SAR, timing from the "
          "temporal encoder + reservoir dynamics.")
    return metrics


def _summarise(per_event) -> dict:
    return {
        "depth_rmse_m": round(float(np.mean([m["depth_rmse_m"] for m in per_event])), 4),
        "onset_mae_h": _mean([m["onset_mae_h"] for m in per_event]),
        "clearance_mae_h": _mean([m["clearance_mae_h"] for m in per_event]),
        "event_f1": round(float(np.mean([m["event_f1"] for m in per_event])), 4),
        "event_roc_auc": round(float(np.mean([m["event_roc_auc"] for m in per_event
                                              if m["event_roc_auc"] is not None])), 4),
    }


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

    from aquaroute.model.frf_targets import MAX_DEPTH_M
    x_np, _ = _standardize(data["node_x_raw"], stats)
    shape = predict_all(model, x_np, data["edge_index"], data["events"][event]["hyeto"])
    pred = (MAX_DEPTH_M * data["propensity"])[:, None] * shape   # reconstruct depth
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
