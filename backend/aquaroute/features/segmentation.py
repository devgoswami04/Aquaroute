"""Road-graph segmentation (Module 2).

Splits OSM edges into uniform ~100 m analysis segments with stable IDs, so every
downstream layer (features, labels, predictions, routing) keys off the same
segment identity. Length-based splitting is done in a metric CRS (UTM 44N) and the
result is reprojected back to EPSG:4326 for storage.

Public function
---------------
segment_road_graph(graph, max_len=100) -> GeoDataFrame[segment_id, geom, ...]
"""
from __future__ import annotations

import math

from aquaroute.config import get_config


def _first(x):
    """Normalise an OSM tag value: unwrap 1-item lists, collapse NaN/empty to None.

    OSM tags are sometimes lists (e.g. highway=['residential','service']) and often
    NaN in the GeoDataFrame — and NaN is truthy in Python, which previously made
    every segment look like an underpass.
    """
    import pandas as pd

    if isinstance(x, list):
        x = x[0] if x else None
    try:
        if x is None or pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass  # non-scalar (shouldn't happen after list unwrap)
    return x


def _is_underpass(row) -> bool:
    tunnel = _first(row.get("tunnel"))
    if tunnel is not None and str(tunnel).lower() not in ("no", "false", "0"):
        return True
    layer = _first(row.get("layer"))
    if layer is not None:
        try:
            if float(layer) < 0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def segment_road_graph(graph, max_len: float | None = None):
    """Return a GeoDataFrame of ~``max_len`` metre road segments (EPSG:4326).

    Columns: ``segment_id, u, v, key, road_class, is_underpass, length_m, geom``.
    """
    import geopandas as gpd
    import osmnx as ox
    from shapely.ops import substring

    cfg = get_config()
    max_len = max_len or float(cfg.features.get("segment_max_len_m", 100))
    metric_crs = cfg.metric_crs

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    edges_m = edges.to_crs(metric_crs)

    records = []
    geoms = []
    for _, row in edges_m.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        total = geom.length  # metres
        n = max(1, int(math.ceil(total / max_len)))
        road_class = _first(row.get("highway"))
        underpass = _is_underpass(row)
        u, v, key = row.get("u"), row.get("v"), row.get("key", 0)
        for i in range(n):
            piece = substring(geom, total * i / n, total * (i + 1) / n)
            if piece.is_empty or piece.length == 0:
                continue
            seg_id = f"{u}_{v}_{key}_{i}"
            records.append({
                "segment_id": seg_id,
                "u": int(u) if u is not None else None,
                "v": int(v) if v is not None else None,
                "key": int(key) if key is not None else 0,
                "road_class": road_class,
                "is_underpass": bool(underpass),
                "length_m": float(piece.length),
            })
            geoms.append(piece)

    seg_m = gpd.GeoDataFrame(records, geometry=geoms, crs=metric_crs)
    seg = seg_m.to_crs(cfg.crs)  # back to EPSG:4326 for storage
    return seg
