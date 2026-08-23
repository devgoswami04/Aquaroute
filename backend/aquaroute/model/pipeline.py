"""Phase 4 pipeline: train + evaluate the susceptibility baseline.

Runs leave-one-event-out validation for XGBoost and RandomForest, prints the
metrics table (the comparison bar for Phase 5), trains a final model on all data,
and saves it plus a metrics JSON. Run with ``python -m aquaroute.model.pipeline``
(or ``make baseline``).
"""
from __future__ import annotations

import json
import sys

from aquaroute.config import get_config
from aquaroute.labels.training_set import assemble_training_set
from aquaroute.model.baseline import save_model, train_baseline
from aquaroute.model.eval_harness import leave_one_event_out, random_split_reference


def _fmt_row(name: str, m: dict) -> str:
    def g(k):
        return f"{m.get(k, float('nan')):.3f}" if k in m else "  -  "
    return (f"  {name:<22} F1={g('f1')}  P={g('precision')}  R={g('recall')}  "
            f"AUC={g('roc_auc')}  PR={g('pr_auc')}  acc={g('accuracy')}")


def run() -> dict:
    cfg = get_config()
    print("AquaRoute Phase 4 — baseline susceptibility model + eval harness\n")

    print("[*] Assembling training set (segments × events)...")
    X, y = assemble_training_set()
    print(f"    rows: {len(X):,}  events: {sorted(X['event'].unique())}  "
          f"flood rate: {y['flooded'].mean():.3f}\n")

    report = {"n_rows": int(len(X)), "results": {}}
    for kind in ("xgboost", "rf"):
        print(f"=== {kind.upper()} — leave-one-event-out ===")
        res = leave_one_event_out(X, y, kind)
        for ev, m in res["per_event"].items():
            print(_fmt_row(f"holdout {ev}", m))
        print(_fmt_row("MEAN (LOEO)", res["mean"]))
        ref = random_split_reference(X, y, kind)
        print(_fmt_row("random-split (optimistic)", ref))
        print("    top features: " + ", ".join(
            f"{k}={v:.2f}" for k, v in list(res["importances"].items())[:5]))
        print()
        report["results"][kind] = {"loeo": res, "random_split": ref}

    # Final model = XGBoost on all data.
    print("[*] Training final XGBoost on all data and saving...")
    final = train_baseline(X, y, "xgboost")
    model_path = cfg.cache_dir / "models" / "baseline_xgb.joblib"
    save_model(final, model_path)
    metrics_path = cfg.cache_dir / "models" / "baseline_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2))
    print(f"    model:   {model_path}")
    print(f"    metrics: {metrics_path}")

    mean_f1 = report["results"]["xgboost"]["loeo"]["mean"].get("f1")
    print(f"\nPhase 4 OK. Baseline bar (XGBoost LOEO mean F1 = {mean_f1}). "
          "Phase 5 FRF must beat this AND add onset/peak/clearance timing.")
    return report


if __name__ == "__main__":
    run()
    sys.exit(0)
