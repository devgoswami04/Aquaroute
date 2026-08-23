"""DEM ingestion (Module 1).

Primary source: **OpenTopography** 30 m global DEM (SRTM GL1), clipped to the bbox
(needs a free key). Keyless fallback: **AWS Terrain Tiles** (Open Data, keyless
GeoTIFF DEM tiles derived from SRTM/others) — a handful of tiles mosaicked and
reprojected to EPSG:4326, so the whole hydrology pipeline (Module 2) runs with no
keys at all. Both write ``data/dem.tif`` with identical interface, so the real
OpenTopography DEM is a drop-in upgrade.

Public function
---------------
fetch_dem(bbox=None, out_tif=None, refresh=False, allow_fallback=True) -> Path
"""
from __future__ import annotations

import math
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from aquaroute.config import BBox, get_config

GLOBALDEM_URL = "https://portal.opentopography.org/API/globaldem"
TERRAIN_TILE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/{z}/{x}/{y}.tif"
_TIMEOUT = 180


def _retry_session() -> requests.Session:
    """Session with exponential backoff that honours Retry-After on 429s."""
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _deg2num(lat: float, lon: float, z: int) -> tuple[int, int]:
    """Slippy-map tile x/y for a lat/lon at zoom z."""
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _fetch_opentopography(bbox: BBox, out: Path, key: str) -> Path:
    dem_type = get_config().ingestion.get("dem_resolution", "SRTMGL1")
    w, s, e, n = bbox.as_west_south_east_north()
    params = {
        "demtype": dem_type,
        "west": w, "south": s, "east": e, "north": n,
        "outputFormat": "GTiff",
        "API_Key": key,
    }
    resp = requests.get(GLOBALDEM_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    out.write_bytes(resp.content)  # GeoTIFF bytes on success
    return out


def _fetch_terrain_tiles(bbox: BBox, out: Path, zoom: int) -> Path:
    """Keyless fallback: mosaic AWS Terrain Tiles over the bbox, reproject to 4326.

    Rasterio is imported lazily so the ingestion package stays importable without
    the geospatial stack for the rainfall-only path.
    """
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

    x0, y0 = _deg2num(bbox.north, bbox.west, zoom)  # top-left tile
    x1, y1 = _deg2num(bbox.south, bbox.east, zoom)  # bottom-right tile
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)

    session = _retry_session()
    memfiles, datasets = [], []
    try:
        for x in xs:
            for y in ys:
                url = TERRAIN_TILE_URL.format(z=zoom, x=x, y=y)
                resp = session.get(url, timeout=_TIMEOUT)
                if resp.status_code == 404:
                    continue  # tile over open sea may be absent
                resp.raise_for_status()
                mf = MemoryFile(resp.content)
                memfiles.append(mf)
                datasets.append(mf.open())
        if not datasets:
            raise RuntimeError("No terrain tiles returned for the bbox.")

        src_crs = datasets[0].crs  # EPSG:3857
        src_nodata = datasets[0].nodata if datasets[0].nodata is not None else -32768.0
        # Clip the mosaic to the bbox (converted to the tile CRS).
        bounds_3857 = transform_bounds("EPSG:4326", src_crs,
                                       bbox.west, bbox.south, bbox.east, bbox.north)
        mosaic, mosaic_tf = merge(datasets, bounds=bounds_3857, nodata=src_nodata)

        # Reproject the clipped mosaic to EPSG:4326 for downstream lon/lat sampling.
        src_h, src_w = mosaic.shape[1], mosaic.shape[2]
        dst_tf, dst_w, dst_h = calculate_default_transform(
            src_crs, "EPSG:4326", src_w, src_h, *bounds_3857)
        dst = np.full((dst_h, dst_w), -9999.0, dtype="float32")
        reproject(
            source=mosaic[0], destination=dst,
            src_transform=mosaic_tf, src_crs=src_crs,
            dst_transform=dst_tf, dst_crs="EPSG:4326",
            src_nodata=src_nodata, dst_nodata=-9999.0, resampling=Resampling.bilinear,
        )
        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "width": dst_w, "height": dst_h, "crs": "EPSG:4326",
            "transform": dst_tf, "nodata": -9999.0,
        }
        with rasterio.open(out, "w", **profile) as f:
            f.write(dst, 1)
    finally:
        for d in datasets:
            d.close()
        for mf in memfiles:
            mf.close()
    return out


def fetch_dem(bbox: BBox | None = None, out_tif: str | Path | None = None,
              refresh: bool = False, allow_fallback: bool = True) -> Path:
    """Fetch a clipped GeoTIFF DEM for the bbox and return its path.

    Uses OpenTopography if a key is set, otherwise the keyless Open-Meteo
    elevation fallback (unless ``allow_fallback=False``, which raises instead).
    """
    cfg = get_config()
    bbox = bbox or cfg.bbox
    out = Path(out_tif) if out_tif else cfg.cache_dir / "dem.tif"

    if out.exists() and not refresh:
        return out

    key = cfg.settings.opentopography_api_key
    if key:
        return _fetch_opentopography(bbox, out, key)

    if not allow_fallback:
        raise RuntimeError(
            "OPENTOPOGRAPHY_API_KEY is not set and allow_fallback=False. Add a free "
            "key from https://portal.opentopography.org/ to .env for the 30 m DEM."
        )

    zoom = int(cfg.ingestion.get("dem_fallback_zoom", 12))
    return _fetch_terrain_tiles(bbox, out, zoom)
