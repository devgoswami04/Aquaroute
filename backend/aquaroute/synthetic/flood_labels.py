"""Synthetic flood ground-truth generator (fallback for GEE SAR).

Produces a binary water-mask GeoTIFF on the DEM grid — the *same* contract as
``ingestion.sar.fetch_sar_flood_mask`` — so the labelling code (Module 3) is
identical whether the mask came from Sentinel-1 or from here. The mask is driven
by terrain (depression depth, TWI, low elevation, flow accumulation) modulated by
the event's real rainfall total, plus seeded noise, so different events flood
different extents.

NOTE: because this is derived from terrain, a model trained purely on these labels
can partly "cheat" via the same terrain features. That circularity is the whole
reason the real SAR path exists — swap it in with a GEE account. The synthetic
path is for making the pipeline runnable and demonstrable end to end.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from aquaroute.config import BBox, get_config
from aquaroute.features.hydrology import HydroLayers


def _seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)


def _norm(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype="float64")
    v = a[valid]
    if v.size == 0:
        return out
    lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
    if hi - lo < 1e-9:
        return out
    out[valid] = np.clip((a[valid] - lo) / (hi - lo), 0, 1)
    return out


def severity_from_rainfall(total_mm: float) -> float:
    """Map an event rainfall total (mm) to the fraction of area that floods."""
    return float(np.clip(0.05 + total_mm / 800.0, 0.05, 0.40))


def synthesize_sar_mask(event_name: str, layers: HydroLayers, out_tif: str | Path,
                        severity: float) -> Path:
    """Write a binary water-mask GeoTIFF for the event on the DEM grid."""
    import rasterio

    valid = np.isfinite(layers.dem) & (layers.dem > -9000)
    score = (
        0.45 * _norm(layers.depression_depth, valid)
        + 0.30 * _norm(layers.twi, valid)
        + 0.25 * _norm(-layers.dem, valid)               # low elevation → wetter
        + 0.20 * _norm(np.log1p(layers.flow_acc), valid)
    )
    rng = np.random.default_rng(_seed(event_name))
    score = score + rng.normal(0, 0.05, size=score.shape)
    score[~valid] = -1.0

    # Threshold so ~`severity` fraction of valid cells become water.
    thr = np.nanpercentile(score[valid], 100 * (1 - severity))
    water = ((score >= thr) & valid).astype("uint8")

    out = Path(out_tif)
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "width": layers.ncols, "height": layers.nrows, "crs": "EPSG:4326",
        "transform": layers.transform, "nodata": 0,
    }
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(water, 1)
    return out


def synthetic_citizen_reports(segments, event_name: str, n: int = 8):
    """A few random citizen reports (flooded/clear) for the merge-labels demo."""
    import pandas as pd

    rng = np.random.default_rng(_seed(event_name + ":reports"))
    ids = segments["segment_id"].sample(min(n, len(segments)),
                                        random_state=int(rng.integers(0, 1_000_000)))
    rows = []
    for sid in ids:
        status = "flooded" if rng.random() < 0.7 else "clear"
        rows.append({
            "segment_id": sid,
            "status": status,
            "depth_est": round(float(rng.uniform(0.1, 0.8)), 2) if status == "flooded" else 0.0,
            "note": f"synthetic citizen report ({event_name})",
        })
    return pd.DataFrame(rows)
