"""Evaluation metrics (Module 4).

Phase 4 covers the classification metrics for the susceptibility baseline. Timing
error (onset/clearance hours) and depth RMSE arrive with the Flood Response
Function in Phase 5 — the baseline has no temporal output, which is exactly the
gap the FRF must fill (brief §8).
"""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Standard binary-classification metrics for flooded vs not."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "n": int(len(y_true)),
        "pos_rate": round(float(y_true.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def mean_metrics(per_event: dict) -> dict:
    """Average scalar metrics across events (macro over held-out events)."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_event.values() if k in m]
        if vals:
            agg[k] = round(float(np.mean(vals)), 4)
    return agg
