"""Per-segment feature vectors and rainfall descriptors (Module 2).

Combines road-segment geometry with the terrain rasters from ``hydrology`` to
produce one feature row per segment: elevation, slope, TWI, depression depth,
upstream contributing area, imperviousness, road class, is_underpass. Also builds
rainfall event descriptors (intensity/duration/antecedent) used by the model.

Imperviousness is a road-class heuristic placeholder until ESA WorldCover land use
is wired in (brief §6 Module 2 / Phase 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aquaroute.features.hydrology import HydroLayers, sample_layers_at

# Road-class → imperviousness fraction (placeholder until WorldCover lands).
_IMPERVIOUS_BY_CLASS = {
    "motorway": 0.95, "trunk": 0.92, "primary": 0.90, "secondary": 0.85,
    "tertiary": 0.80, "residential": 0.65, "living_street": 0.60,
    "unclassified": 0.55, "service": 0.50,
}


def _imperviousness(road_class) -> float:
    return _IMPERVIOUS_BY_CLASS.get(str(road_class), 0.6)


def build_segment_features(segments, layers: HydroLayers, landuse=None):
    """Attach terrain + surface features to the segments GeoDataFrame.

    Returns the same GeoDataFrame with added feature columns and a heuristic
    ``susceptibility`` score in [0, 1] (a pre-model stand-in for map colouring;
    replaced by real predictions from Module 4 in Phase 5/6).
    """
    seg = segments.copy()
    centroids = seg.geometry.centroid
    feats = []
    for pt in centroids:
        feats.append(sample_layers_at(layers, pt.x, pt.y))
    fdf = pd.DataFrame(feats, index=seg.index)

    seg["elevation"] = fdf["elevation"]
    seg["slope"] = fdf["slope"]
    seg["twi"] = fdf["twi"]
    seg["depression_depth"] = fdf["depression_depth"]
    seg["upstream_area"] = fdf["upstream_area"]
    seg["imperviousness"] = seg["road_class"].map(_imperviousness)

    seg["susceptibility"] = _static_susceptibility(seg)
    return seg


def _static_susceptibility(seg) -> "pd.Series":
    """Heuristic flood susceptibility from terrain — NOT the trained model.

    High TWI, high depression depth, low elevation and underpasses raise risk.
    Min-max normalised across the corridor into [0, 1].
    """
    def norm(s: pd.Series) -> pd.Series:
        s = s.astype("float64")
        lo, hi = s.min(skipna=True), s.max(skipna=True)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - lo) / (hi - lo)

    twi_n = norm(seg["twi"])
    depr_n = norm(seg["depression_depth"])
    elev_n = 1.0 - norm(seg["elevation"])  # lower elevation = higher risk
    score = 0.4 * twi_n + 0.35 * depr_n + 0.25 * elev_n
    score = score + seg["is_underpass"].astype(float) * 0.15  # underpasses flood first
    return score.clip(0.0, 1.0).fillna(0.0)


def build_rainfall_descriptors(rain: pd.DataFrame, event: str = "") -> pd.DataFrame:
    """Summarise a rainfall series into per-grid-point event descriptors.

    Expects the tidy rainfall frame from ``ingestion.rainfall`` (hourly rows).
    Returns intensity (max mm/h), total depth, duration (wet hours), antecedent
    wetness (first-6h total) and a simple hyetograph peak-time fraction.
    """
    hourly = rain[rain.get("resolution", "hourly") == "hourly"].copy()
    if hourly.empty:
        return pd.DataFrame(columns=[
            "point_id", "event", "intensity_max", "total_mm", "duration_wet_h",
            "antecedent_6h", "peak_time_frac"])
    out = []
    for pid, grp in hourly.sort_values("time").groupby("point_id"):
        precip = grp["precip_mm"].fillna(0.0).to_numpy()
        n = len(precip)
        wet = int((precip > 0.1).sum())
        peak_idx = int(np.argmax(precip)) if n else 0
        out.append({
            "point_id": pid,
            "event": event,
            "intensity_max": float(precip.max()) if n else 0.0,
            "total_mm": float(precip.sum()),
            "duration_wet_h": wet,
            "antecedent_6h": float(precip[:6].sum()),
            "peak_time_frac": (peak_idx / n) if n else 0.0,
        })
    return pd.DataFrame(out)
