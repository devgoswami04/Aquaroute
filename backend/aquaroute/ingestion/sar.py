"""Sentinel-1 SAR flood-mask ingestion via Google Earth Engine (Module 1).

Water is dark in SAR (low backscatter), so a flood shows up as a drop in VV
backscatter relative to a dry reference. This is the standard S1 change-detection
approach. Requires a free GEE research account: run ``earthengine authenticate``
once and set ``EE_PROJECT`` in .env. When GEE isn't configured the labelling
pipeline falls back to the synthetic mask generator (identical GeoTIFF contract).

Public function
---------------
fetch_sar_flood_mask(event_date, bbox=None, out_tif=None, window_days=6) -> Path
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from aquaroute.config import BBox, get_config


def _ee_init():
    import ee  # lazy; only needed on the real GEE path

    cfg = get_config()
    project = cfg.settings.ee_project
    if not project:
        raise RuntimeError(
            "EE_PROJECT is not set. Run `earthengine authenticate` and set EE_PROJECT "
            "in .env, or let the pipeline use the synthetic fallback."
        )
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def fetch_sar_flood_mask(event_date: str, bbox: BBox | None = None,
                         out_tif: str | Path | None = None,
                         window_days: int = 6, threshold_db: float = -17.0) -> Path:
    """Download a binary SAR water mask GeoTIFF around ``event_date`` (YYYY-MM-DD).

    Uses Sentinel-1 GRD VV: a min-composite over the event window thresholded at
    ``threshold_db`` dB. Exported via ``getDownloadURL`` clipped to the bbox.
    """
    import zipfile
    import io
    import requests

    ee = _ee_init()
    cfg = get_config()
    bbox = bbox or cfg.bbox
    out = Path(out_tif) if out_tif else cfg.cache_dir / f"sar_{event_date}.tif"

    d = datetime.fromisoformat(event_date)
    start = (d - timedelta(days=window_days)).date().isoformat()
    end = (d + timedelta(days=window_days)).date().isoformat()
    geom = ee.Geometry.Rectangle([bbox.west, bbox.south, bbox.east, bbox.north])

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(geom)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .select("VV"))
    flood = s1.filterDate(start, end).min()          # min backscatter = wettest
    water = flood.lt(threshold_db).rename("water").toByte().clip(geom)

    url = water.getDownloadURL({
        "scale": 30, "region": geom, "format": "GEO_TIFF",
        "crs": "EPSG:4326",
    })
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    content = resp.content
    # getDownloadURL may return a zip or a bare tif depending on options.
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = next(n for n in z.namelist() if n.endswith(".tif"))
            out.write_bytes(z.read(name))
    else:
        out.write_bytes(content)
    return out
