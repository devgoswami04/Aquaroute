"""Road-graph ingestion from OpenStreetMap via OSMnx (Module 1).

Downloads the drivable road network for the study bbox as a ready-made NetworkX
MultiDiGraph with geometry and edge lengths, and caches it to
``data/road_graph.graphml`` so repeat runs are offline and reproducible.

Public function
---------------
fetch_road_graph(place_or_polygon=None, refresh=False) -> networkx.MultiDiGraph
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aquaroute.config import get_config

if TYPE_CHECKING:  # import only for type checkers; keeps rainfall usable standalone
    import networkx as nx


def _graph_cache_path() -> Path:
    return get_config().cache_dir / "road_graph.graphml"


def fetch_road_graph(place_or_polygon=None, refresh: bool = False) -> "nx.MultiDiGraph":
    """Return the drivable OSM graph for the study area, cached to GraphML.

    Parameters
    ----------
    place_or_polygon
        Optional override. If a string, treated as an OSMnx place query; if a
        shapely Polygon, used directly. Default: the configured bbox (brief §1).
    refresh
        Re-download even if a cached GraphML exists.
    """
    import osmnx as ox  # imported lazily — heavy geospatial import

    cfg = get_config()
    cache = _graph_cache_path()

    if cache.exists() and not refresh:
        return ox.load_graphml(cache)

    if place_or_polygon is None:
        n, s, e, w = cfg.bbox.as_osmnx()
        # OSMnx >=2.0 takes a (west, south, east, north) tuple; <2.0 takes kwargs.
        try:
            graph = ox.graph_from_bbox(
                bbox=(w, s, e, n), network_type=cfg.ingestion.get("road_network_type", "drive")
            )
        except TypeError:  # older OSMnx signature
            graph = ox.graph_from_bbox(
                north=n, south=s, east=e, west=w,
                network_type=cfg.ingestion.get("road_network_type", "drive"),
            )
    elif isinstance(place_or_polygon, str):
        graph = ox.graph_from_place(
            place_or_polygon, network_type=cfg.ingestion.get("road_network_type", "drive")
        )
    else:  # assume a shapely polygon
        graph = ox.graph_from_polygon(
            place_or_polygon, network_type=cfg.ingestion.get("road_network_type", "drive")
        )

    ox.save_graphml(graph, cache)
    return graph
