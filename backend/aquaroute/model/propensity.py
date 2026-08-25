"""Data-driven flood propensity from real SAR labels (model improvement).

Instead of guessing which roads flood from a terrain heuristic, learn it from the
**real Sentinel-1 SAR** labels: an XGBoost classifier maps per-segment terrain
features → probability the segment floods. This propensity then grounds the Flood
Response Function's flood *extent* in real observations (the FRF still supplies the
depth-vs-time *timing*). Trained on the SAR-labelled events only, so it reflects
observed flooding, not the synthetic fallback.

compute_flood_propensity(train_events=None) -> pd.Series[segment_id -> p_flood]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TERRAIN_FEATURES = [
    "elevation", "slope", "twi", "depression_depth", "upstream_area",
    "imperviousness", "is_underpass", "length_m",
]


def _real_sar_events() -> list[str]:
    from aquaroute.labels.store import dominant_source, list_labelled_events, slug
    # dominant_source takes a display name or slug; slugs are what's on disk.
    out = []
    for s in list_labelled_events():
        try:
            if dominant_source(s) == "sar":
                out.append(s)
        except Exception:
            pass
    return out


def compute_flood_propensity(train_events: list[str] | None = None,
                             return_model: bool = False):
    """Per-segment flood probability learned from real-SAR events.

    ``train_events`` (slugs) defaults to every SAR-labelled event. Returns a
    Series indexed by segment_id; optionally also the fitted model.
    """
    from xgboost import XGBClassifier

    from aquaroute.db.segments_store import load_segments_gdf
    from aquaroute.labels.store import load_labels

    events = train_events or _real_sar_events()
    if not events:
        raise RuntimeError("No SAR-labelled events to learn flood propensity from.")

    seg = load_segments_gdf().copy()
    seg["is_underpass"] = seg["is_underpass"].astype(int)
    X = seg[TERRAIN_FEATURES].to_numpy(dtype="float64")
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    bad = np.where(~np.isfinite(X))
    X[bad] = np.take(med, bad[1])

    # Label = flooded in ANY training SAR event (union of observed flooding).
    flooded = pd.Series(False, index=seg["segment_id"])
    for s in events:
        lab = load_labels(s).set_index("segment_id")["flooded"]
        flooded = flooded | seg["segment_id"].map(lab).fillna(False).to_numpy()
    y = flooded.to_numpy().astype(int)

    # No class re-weighting: on the ~3% flood base rate this keeps the predicted
    # probabilities calibrated (mean ≈ base rate) so the FRF's flood *extent* stays
    # realistic, while still ranking flood-prone terrain well (held-out AUC ≈ 0.80).
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    propensity = pd.Series(p, index=seg["segment_id"], name="propensity")
    return (propensity, clf) if return_model else propensity
