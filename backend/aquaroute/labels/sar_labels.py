"""Turn a SAR/synthetic water mask into per-segment flood labels (Module 3).

Samples the water raster along each road segment; a segment is labelled flooded
when enough of its samples fall on water. A depth proxy is read from the DEM's
depression depth at the segment (SAR gives presence, not depth), which is a
sensible stand-in for how deep water can pool there.

Public function
---------------
label_event_from_sar(mask_tif, segments, layers=None, water_frac=0.3, source='sar')
    -> DataFrame[segment_id, flooded, depth_proxy, source]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sample_grid(lons: np.ndarray, lats: np.ndarray, array: np.ndarray, transform):
    """Nearest-cell sample of a north-up raster at vectorised lon/lat points."""
    a, e, c, f = transform.a, transform.e, transform.c, transform.f
    cols = ((lons - c) / a).astype(int)
    rows = ((lats - f) / e).astype(int)
    h, w = array.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    vals = np.full(lons.shape, np.nan, dtype="float64")
    vals[valid] = array[rows[valid], cols[valid]]
    return vals, valid


def label_event_from_sar(mask_tif: str | Path, segments, layers=None,
                         water_frac: float = 0.3, n_samples: int = 3,
                         source: str = "sar") -> pd.DataFrame:
    """Label each segment flooded/not from the water mask, with a depth proxy."""
    import rasterio

    with rasterio.open(mask_tif) as src:
        mask = src.read(1)
        mtf = src.transform
        mnodata = src.nodata

    # Sample points spread along each segment (normalised distances).
    fracs = np.linspace(0.15, 0.85, n_samples)
    geom = segments.geometry
    water_hits = np.zeros(len(segments), dtype="float64")
    valid_counts = np.zeros(len(segments), dtype="float64")
    for fr in fracs:
        pts = geom.interpolate(fr, normalized=True)
        lons = pts.x.to_numpy()
        lats = pts.y.to_numpy()
        vals, valid = _sample_grid(lons, lats, mask, mtf)
        is_water = valid & (vals > 0) & (~np.isclose(vals, mnodata if mnodata is not None else -1))
        water_hits += is_water.astype(float)
        valid_counts += valid.astype(float)

    frac = np.divide(water_hits, np.maximum(valid_counts, 1))
    flooded = frac >= water_frac

    # Depth proxy from DEM depression depth at the segment centroid.
    if layers is not None:
        cen = geom.centroid
        depr, dvalid = _sample_grid(cen.x.to_numpy(), cen.y.to_numpy(),
                                    layers.depression_depth, layers.transform)
        depth_proxy = np.where(flooded, np.clip(np.nan_to_num(depr), 0.05, None), 0.0)
    else:
        depth_proxy = np.where(flooded, 0.2 + 0.6 * frac, 0.0)

    return pd.DataFrame({
        "segment_id": segments["segment_id"].to_numpy(),
        "flooded": flooded,
        "depth_proxy": np.round(depth_proxy, 3),
        "source": source,
    })
