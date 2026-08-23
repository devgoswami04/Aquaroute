"""Assemble the (X, y) training set from segment features + event labels (Module 3).

X = per-segment terrain/surface features joined with per-event rainfall descriptors
(bbox-mean, broadcast to every segment). y = flooded label + depth proxy. This is
the input to the Phase 4 baseline and, later, the Flood Response Function.
"""
from __future__ import annotations

import pandas as pd

from aquaroute.config import get_config

_FEATURE_COLS = [
    "length_m", "elevation", "slope", "twi", "depression_depth",
    "upstream_area", "imperviousness", "is_underpass",
]
_RAIN_COLS = ["intensity_max", "total_mm", "duration_wet_h", "antecedent_6h", "peak_time_frac"]


def _event_rain_descriptors(start: str, end: str) -> dict:
    """Mean rainfall descriptors over the bbox for an event window (best-effort)."""
    try:
        from aquaroute.features.segment_features import build_rainfall_descriptors
        from aquaroute.ingestion.rainfall import fetch_rainfall_history

        rain = fetch_rainfall_history(start=start, end=end)
        desc = build_rainfall_descriptors(rain)
        return {c: float(desc[c].mean()) for c in _RAIN_COLS if c in desc}
    except Exception:
        return {c: 0.0 for c in _RAIN_COLS}


def assemble_training_set(events: list[str] | None = None):
    """Return (X, y) DataFrames indexed identically over segment×event rows."""
    from aquaroute.db.segments_store import load_segments_gdf
    from aquaroute.labels.store import list_labelled_events, load_labels

    cfg = get_config()
    seg = load_segments_gdf().drop(columns="geometry")
    seg_feats = seg[["segment_id"] + [c for c in _FEATURE_COLS if c in seg.columns]].copy()
    seg_feats["is_underpass"] = seg_feats["is_underpass"].astype(int)

    labelled = list_labelled_events()
    ev_by_slug = {}
    for e in cfg.events:
        from aquaroute.labels.store import slug
        ev_by_slug[slug(e["name"])] = e

    X_parts, y_parts = [], []
    for s in labelled:
        if events and s not in events:
            continue
        labels = load_labels(s)
        meta = ev_by_slug.get(s, {})
        rain = _event_rain_descriptors(meta.get("start", ""), meta.get("end", ""))

        df = seg_feats.merge(labels[["segment_id", "flooded", "depth_proxy"]], on="segment_id")
        for c in _RAIN_COLS:
            df[c] = rain.get(c, 0.0)
        df["event"] = s
        X_parts.append(df[["segment_id", "event"] + _FEATURE_COLS + _RAIN_COLS])
        y_parts.append(df[["flooded", "depth_proxy"]])

    if not X_parts:
        raise RuntimeError("No labelled events found. Run the Phase 3 labels pipeline first.")
    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    y["flooded"] = y["flooded"].astype(int)
    return X, y
