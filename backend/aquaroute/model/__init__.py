"""Module 4 — model core.

Phase 4: RF/XGBoost susceptibility baseline + evaluation harness. Phase 5 adds the
novel Flood Response Function (TCN/LSTM temporal encoder + PyG GNN → depth-vs-time).
"""
from aquaroute.model.baseline import (
    FEATURES,
    feature_importances,
    load_model,
    save_model,
    train_baseline,
)
from aquaroute.model.evaluate import classification_metrics
from aquaroute.model.eval_harness import leave_one_event_out

__all__ = [
    "FEATURES",
    "train_baseline",
    "save_model",
    "load_model",
    "feature_importances",
    "classification_metrics",
    "leave_one_event_out",
]


def __getattr__(name):
    # Lazy re-exports so importing the baseline doesn't require torch/PyG.
    if name in ("FloodResponseFunction", "predict_curve", "derive_events", "predict_all"):
        import aquaroute.model.frf as _frf
        return getattr(_frf, name)
    raise AttributeError(name)
