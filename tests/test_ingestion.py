"""Phase 1 tests.

Network-dependent tests are marked so they can be skipped offline
(``pytest -m "not network"``). The rainfall test prints the 24 h series and the
road-graph test prints the segment count — the Phase 1 verification the brief asks
for (§10, §12).
"""
from __future__ import annotations

import os

import pytest

from aquaroute.config import get_config
from aquaroute.ingestion.rainfall import _grid_points

network = pytest.mark.skipif(
    os.environ.get("AQUAROUTE_SKIP_NETWORK") == "1",
    reason="network disabled via AQUAROUTE_SKIP_NETWORK=1",
)


def test_config_bbox_orderings():
    """BBox helper returns library-correct orderings (no lat/lon swaps)."""
    cfg = get_config()
    n, s, e, w = cfg.bbox.as_osmnx()
    assert n > s and e > w
    w2, s2, e2, n2 = cfg.bbox.as_west_south_east_north()
    assert (w2, s2, e2, n2) == (w, s, e, n)


def test_grid_points_lattice():
    cfg = get_config()
    pts = _grid_points(cfg.bbox, n=3)
    assert len(pts) == 9
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    assert min(lats) >= cfg.bbox.south and max(lats) <= cfg.bbox.north
    assert min(lons) >= cfg.bbox.west and max(lons) <= cfg.bbox.east


@network
def test_rainfall_forecast_24h_series(capsys):
    from aquaroute.ingestion.rainfall import fetch_rainfall_forecast

    rain = fetch_rainfall_forecast(hours=24)
    hourly = rain[rain["resolution"] == "hourly"]
    assert not hourly.empty
    # one 24-point series per grid point
    per_point = hourly.groupby("point_id").size()
    assert (per_point <= 24).all() and per_point.max() >= 20

    centre = hourly["point_id"].iloc[0]
    series = hourly[hourly["point_id"] == centre].sort_values("time")
    print("\n24 h precipitation (mm) at", centre)
    for _, r in series.iterrows():
        print(f"  {r['time']:%Y-%m-%d %H:%M}  {r['precip_mm']:.2f}")
    assert series["precip_mm"].notna().any()


@network
def test_road_graph_segment_count(capsys):
    from aquaroute.ingestion.roads import fetch_road_graph

    graph = fetch_road_graph()
    n_edges = graph.number_of_edges()
    print(f"\nRoad graph: {graph.number_of_nodes()} nodes, {n_edges} edges (segments)")
    assert n_edges > 100  # the corridor has thousands of drivable edges
