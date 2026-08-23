"""Evaluation harness (Module 4) — leave-one-event-out validation.

§8 requires holding out a whole monsoon event for validation. With three events we
run full leave-one-event-out: train on the other two, test on the held-out one.
This measures whether the model generalises the terrain→flood mapping to an event
with a different rainfall severity — a much harder and more honest test than a
random within-event split.
"""
from __future__ import annotations

from aquaroute.model.baseline import FEATURES, feature_importances, train_baseline
from aquaroute.model.evaluate import classification_metrics, mean_metrics


def leave_one_event_out(X, y, kind: str = "xgboost") -> dict:
    """Return {per_event, mean, importances} for a LOEO run."""
    events = list(X["event"].unique())
    per_event = {}
    for ev in events:
        tr = X["event"] != ev
        te = ~tr
        model = train_baseline(X[tr], y[tr], kind)
        prob = model.predict_proba(X.loc[te, FEATURES])[:, 1]
        pred = (prob >= 0.5).astype(int)
        per_event[ev] = classification_metrics(y.loc[te, "flooded"], pred, prob)

    # Importances from a model trained on everything (for reporting).
    full = train_baseline(X, y, kind)
    return {
        "kind": kind,
        "per_event": per_event,
        "mean": mean_metrics(per_event),
        "importances": feature_importances(full),
    }


def random_split_reference(X, y, kind: str = "xgboost", test_size: float = 0.2) -> dict:
    """Optimistic within-distribution baseline (random split) for contrast."""
    from sklearn.model_selection import train_test_split

    idx = X.index.to_numpy()
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=42,
                                      stratify=y["flooded"])
    model = train_baseline(X.loc[tr_idx], y.loc[tr_idx], kind)
    prob = model.predict_proba(X.loc[te_idx, FEATURES])[:, 1]
    pred = (prob >= 0.5).astype(int)
    return classification_metrics(y.loc[te_idx, "flooded"], pred, prob)
