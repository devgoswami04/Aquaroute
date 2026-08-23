"""Phase 4 tests — baseline training, metrics, and the eval harness.

The core tests build a tiny synthetic dataset so they run fast and always. A
skip-guarded test checks the saved model artifact if the pipeline has been run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "data" / "models" / "baseline_xgb.joblib"


def _toy(n=400, seed=0):
    """Separable-ish toy set on the real FEATURES columns, across 3 events."""
    from aquaroute.model.baseline import FEATURES

    rng = np.random.default_rng(seed)
    rows, ys, evs = [], [], []
    for ei, ev in enumerate(["e1", "e2", "e3"]):
        for _ in range(n):
            twi = rng.uniform(3, 22)
            depr = rng.uniform(0, 10)
            elev = rng.uniform(0, 60)
            score = 0.5 * twi / 22 + 0.5 * depr / 10 - 0.4 * elev / 60 + 0.1 * ei
            flooded = int(score + rng.normal(0, 0.1) > 0.4)
            feat = {c: 0.0 for c in FEATURES}
            feat.update(twi=twi, depression_depth=depr, elevation=elev,
                        total_mm=100 + 80 * ei, length_m=55.0, imperviousness=0.7)
            rows.append(feat); ys.append(flooded); evs.append(ev)
    X = pd.DataFrame(rows); X["event"] = evs; X["segment_id"] = range(len(X))
    y = pd.DataFrame({"flooded": ys, "depth_proxy": np.where(np.array(ys) == 1, 0.3, 0.0)})
    return X, y


def test_train_and_predict():
    from aquaroute.model.baseline import FEATURES, train_baseline

    X, y = _toy()
    model = train_baseline(X, y, "xgboost")
    prob = model.predict_proba(X[FEATURES])[:, 1]
    assert prob.shape[0] == len(X)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_classification_metrics_keys():
    from aquaroute.model.evaluate import classification_metrics

    m = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    for k in ("accuracy", "precision", "recall", "f1", "confusion", "roc_auc"):
        assert k in m


def test_leave_one_event_out_runs():
    from aquaroute.model.eval_harness import leave_one_event_out

    X, y = _toy(n=250)
    res = leave_one_event_out(X, y, "xgboost")
    assert set(res["per_event"]) == {"e1", "e2", "e3"}
    assert "f1" in res["mean"]
    assert res["importances"]  # non-empty importance dict


@pytest.mark.skipif(not MODEL.exists(), reason="baseline model not trained")
def test_saved_model_predicts():
    from aquaroute.model.baseline import FEATURES, load_model

    model = load_model(MODEL)
    X, _ = _toy(n=10)
    pred = model.predict(X[FEATURES])
    assert len(pred) == len(X)
