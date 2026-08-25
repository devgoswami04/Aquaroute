"""Sentinel-1 SAR flood-mask ingestion (Module 1).

Real flood ground truth from Sentinel-1 C-band SAR. Water is specular (dark) in
SAR, so a flood shows up as a **drop in VV backscatter** relative to a dry
reference — the standard change-detection method (UN-SPIDER / Copernicus EMS).

Two sources, tried in order:
  1. **Microsoft Planetary Computer** STAC (`sentinel-1-grd`) — *keyless*
     anonymous signed reads. Primary path; no account needed.
  2. **Google Earth Engine** (`COPERNICUS/S1_GRD`) — needs a free research account
     (`earthengine authenticate` + EE_PROJECT). Fallback for dates PC lacks.

Raises ``SarUnavailable`` when neither has a scene covering the event window (e.g.
a 12-day revisit gap), so the labelling pipeline can fall back to the synthetic
generator for that specific event.

Public function
---------------
fetch_sar_flood_mask(event_start, event_end, bbox=None, out_tif=None, ...) -> Path
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from aquaroute.config import BBox, get_config

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
_RES_DEG = 0.0002              # ~20 m grid for the flood mask
CHANGE_DB = 3.0                # backscatter drop (dB) that signals flooding
NODATA = 0


class SarUnavailable(RuntimeError):
    """No SAR scene covers the event window in any configured source."""


def _iso(d) -> str:
    return d if isinstance(d, str) else d.isoformat()


# --------------------------------------------------------------------------- #
# Planetary Computer path (keyless)
# --------------------------------------------------------------------------- #
def _pc_catalog():
    import planetary_computer as pc
    import pystac_client
    return pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)

def _pc_scenes(cat, bbox: BBox, start: str, end: str):
    w, s, e, n = bbox.as_west_south_east_north()
    items = list(cat.search(collections=["sentinel-1-grd"],
                            bbox=[w, s, e, n], datetime=f"{start}/{end}").items())
    return sorted(items, key=lambda it: it.datetime)


def _pc_load_vv_db(item, bbox: BBox) -> tuple[np.ndarray, object]:
    """Load a scene's VV band over the bbox, return (dB array, rio transform)."""
    import odc.stac
    w, s, e, n = bbox.as_west_south_east_north()
    ds = odc.stac.load([item], bands=["vv"], bbox=[w, s, e, n],
                       crs="EPSG:4326", resolution=_RES_DEG)
    vv = ds["vv"].isel(time=0)
    arr = vv.values.astype("float64")
    # GRD pixels are amplitude DN; dB(sigma0) = 20*log10(DN) + K (K cancels in the
    # change ratio). Guard non-positive DN.
    db = 20.0 * np.log10(np.clip(arr, 1.0, None))
    db[arr <= 0] = np.nan
    return db, ds.odc.geobox.transform


def detect_water(dur_db: np.ndarray, ref_db: np.ndarray, change_db: float = CHANGE_DB,
                 abs_pct: float = 8.0) -> np.ndarray:
    """Change-detection flood water mask from during/reference VV dB arrays.

    Water = backscatter dropped >= ``change_db`` vs the dry reference (new
    flooding) OR very dark now (permanent/standing water, from the dark tail).
    Median-filtered to suppress speckle and morphologically opened to drop
    isolated pixels. Pure function (no I/O) so it is unit-testable.
    """
    from scipy.ndimage import binary_opening, median_filter

    valid = np.isfinite(dur_db) & np.isfinite(ref_db)
    dur_s = median_filter(np.nan_to_num(dur_db, nan=0.0), size=5)
    ref_s = median_filter(np.nan_to_num(ref_db, nan=0.0), size=5)
    change = ref_s - dur_s
    abs_water = np.nanpercentile(dur_s[valid], abs_pct) if valid.any() else -np.inf
    water = valid & ((change > change_db) | (dur_s < abs_water))
    return binary_opening(water, iterations=1)


def _fetch_pc(bbox: BBox, out: Path, start: str, end: str) -> Path:
    cat = _pc_catalog()
    es, ee = date.fromisoformat(start), date.fromisoformat(end)
    during = _pc_scenes(cat, bbox, (es - timedelta(days=1)).isoformat(),
                        (ee + timedelta(days=10)).isoformat())
    reference = _pc_scenes(cat, bbox, (es - timedelta(days=90)).isoformat(),
                           (es - timedelta(days=18)).isoformat())
    if not during or not reference:
        raise SarUnavailable(
            f"Planetary Computer has no during+reference S1 pair for {start}..{end}")

    dur_item = during[-1]        # closest after the event (water still present)
    ref_item = reference[-1]     # most recent dry scene before the event
    dur_db, transform = _pc_load_vv_db(dur_item, bbox)
    ref_db, _ = _pc_load_vv_db(ref_item, bbox)

    # Align shapes (scenes can differ by a row/col).
    h = min(dur_db.shape[0], ref_db.shape[0]); w_ = min(dur_db.shape[1], ref_db.shape[1])
    dur_db, ref_db = dur_db[:h, :w_], ref_db[:h, :w_]

    water = detect_water(dur_db, ref_db, CHANGE_DB)
    _write_mask(water.astype("uint8"), transform, out)
    _log_provenance(out, "planetary-computer", dur_item.id, ref_item.id,
                    dur_item.datetime.date().isoformat(), ref_item.datetime.date().isoformat(),
                    float(water.mean()))
    return out


# --------------------------------------------------------------------------- #
# Google Earth Engine path (needs a free account)
# --------------------------------------------------------------------------- #
def _fetch_gee(bbox: BBox, out: Path, start: str, end: str,
               window_days: int = 6, threshold_db: float = -17.0) -> Path:
    import io
    import zipfile

    import ee
    import requests

    project = get_config().settings.ee_project
    if not project:
        raise SarUnavailable("EE_PROJECT not set; GEE path unavailable")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(); ee.Initialize(project=project)

    geom = ee.Geometry.Rectangle([bbox.west, bbox.south, bbox.east, bbox.north])
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(geom).filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).select("VV"))
    flood = s1.filterDate(start, end).min()
    water = flood.lt(threshold_db).rename("water").toByte().clip(geom)
    url = water.getDownloadURL({"scale": 30, "region": geom, "format": "GEO_TIFF", "crs": "EPSG:4326"})
    content = requests.get(url, timeout=300).content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            out.write_bytes(z.read(next(n for n in z.namelist() if n.endswith(".tif"))))
    else:
        out.write_bytes(content)
    return out


# --------------------------------------------------------------------------- #
def _write_mask(mask: np.ndarray, transform, out: Path) -> None:
    import rasterio
    h, w = mask.shape
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1, "height": h, "width": w,
               "crs": "EPSG:4326", "transform": transform, "nodata": NODATA}
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(mask, 1)


def _log_provenance(out: Path, source, dur_id, ref_id, dur_date, ref_date, frac) -> None:
    meta = out.with_suffix(".json")
    import json
    meta.write_text(json.dumps({
        "source": source, "during_scene": dur_id, "reference_scene": ref_id,
        "during_date": dur_date, "reference_date": ref_date,
        "water_fraction": round(frac, 4),
    }, indent=2))


def fetch_sar_flood_mask(event_start: str, event_end: str, bbox: BBox | None = None,
                         out_tif: str | Path | None = None,
                         prefer: str = "pc") -> Path:
    """Real Sentinel-1 flood water mask (GeoTIFF) for an event window.

    Tries Planetary Computer (keyless) then GEE. Raises ``SarUnavailable`` if
    neither has a scene, so the caller can fall back to the synthetic generator.
    """
    cfg = get_config()
    bbox = bbox or cfg.bbox
    out = Path(out_tif) if out_tif else cfg.cache_dir / f"sar_{_iso(event_start)}.tif"
    if out.exists():
        return out

    order = [_fetch_pc, _fetch_gee] if prefer == "pc" else [_fetch_gee, _fetch_pc]
    last: Exception | None = None
    for fn in order:
        try:
            return fn(bbox, out, _iso(event_start), _iso(event_end))
        except SarUnavailable as e:
            last = e
        except Exception as e:               # unexpected (network, auth) — try next
            last = e
    raise SarUnavailable(f"No SAR scene for {event_start}..{event_end}: {last}")
