"""Persist & read the segments layer.

Authoritative store is PostGIS when available; a GeoJSON file cache
(``data/segments.geojson``) is always written so the map and ``/segments`` work
without the database container running.
"""
from __future__ import annotations

import json
from pathlib import Path

from aquaroute.config import get_config
from aquaroute.db.session import get_engine, postgis_available

TABLE = "segments"


def _cache_path() -> Path:
    # GeoParquet, not GeoJSON: 121k features read in ~1 s vs ~90 s for fiona JSON.
    return get_config().cache_dir / "segments.parquet"


def save_segments(seg_gdf) -> dict:
    """Write segments to the file cache and (if reachable) PostGIS.

    Returns a small summary dict {count, postgis, file}.
    """
    path = _cache_path()
    seg_gdf.to_parquet(path)

    wrote_pg = False
    if postgis_available():
        from aquaroute.db.session import init_db

        init_db()  # ensure PostGIS extension + tables
        seg_gdf.to_postgis(TABLE, get_engine(), if_exists="replace", index=False)
        wrote_pg = True

    return {"count": int(len(seg_gdf)), "postgis": wrote_pg, "file": str(path)}


_GDF_CACHE = {}


def load_segments_gdf():
    """Read segments as a GeoDataFrame (PostGIS first, else file cache).

    Cached in-memory so repeated /segments requests don't re-read 121k features.
    """
    import geopandas as gpd

    if "gdf" in _GDF_CACHE:
        return _GDF_CACHE["gdf"]

    gdf = None
    if postgis_available():
        try:
            gdf = gpd.read_postgis(f"SELECT * FROM {TABLE}", get_engine(), geom_col="geom")
        except Exception:
            gdf = None  # table not built yet → fall back to file
    if gdf is None:
        path = _cache_path()
        if not path.exists():
            raise FileNotFoundError(
                "No segments found. Run Phase 2: `python -m aquaroute.features.pipeline`."
            )
        gdf = gpd.read_parquet(path)

    if gdf.geometry.name != "geometry":
        gdf = gdf.rename_geometry("geometry")
    _GDF_CACHE["gdf"] = gdf
    return gdf


def load_segments_geojson(classes: list[str] | None = None,
                          bbox: tuple[float, float, float, float] | None = None,
                          limit: int | None = None,
                          precision: int = 5) -> dict:
    """Return the segments layer as a GeoJSON FeatureCollection dict.

    ``classes`` filters by road_class; ``bbox`` = (west, south, east, north)
    spatial filter; ``limit`` caps the feature count; ``precision`` rounds
    coordinates to N decimals (~1 m at 5) to shrink the payload for the browser.
    """
    from shapely import set_precision

    gdf = load_segments_gdf()
    if classes:
        gdf = gdf[gdf["road_class"].astype(str).isin(classes)]
    if bbox:
        w, s, e, n = bbox
        gdf = gdf.cx[w:e, s:n]
    if limit is not None:
        gdf = gdf.iloc[:limit]

    gdf = gdf.copy()
    grid = 10 ** (-precision)
    gdf["geometry"] = set_precision(gdf.geometry.values, grid_size=grid)
    return json.loads(gdf.to_json())


def clear_cache() -> None:
    _GDF_CACHE.clear()
