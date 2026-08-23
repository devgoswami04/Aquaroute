"""Phase 2 tests — segmentation, hydrology, and the /segments endpoint.

The heavy hydrology/segmentation tests require the cached artifacts produced by
``python -m aquaroute.features.pipeline`` (DEM + road graph). They skip cleanly if
those aren't present, so the suite still runs on a fresh checkout.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEM = REPO / "data" / "dem.tif"
GRAPH = REPO / "data" / "road_graph.graphml"
SEGMENTS = REPO / "data" / "segments.parquet"


@pytest.mark.skipif(not DEM.exists(), reason="DEM not fetched (run features pipeline)")
def test_condition_dem_produces_layers():
    from aquaroute.features.hydrology import condition_dem

    layers = condition_dem(DEM)
    assert layers.nrows > 1 and layers.ncols > 1
    assert layers.flow_acc.min() >= 0
    # Depression depth is non-negative by construction (filled >= original).
    assert float(layers.depression_depth.min()) >= -1e-6
    assert layers.twi.shape == layers.dem.shape


@pytest.mark.skipif(not GRAPH.exists(), reason="road graph not cached")
def test_segmentation_lengths_and_ids():
    from aquaroute.features.segmentation import segment_road_graph
    from aquaroute.ingestion.roads import fetch_road_graph

    seg = segment_road_graph(fetch_road_graph())
    assert len(seg) > 1000
    assert seg["segment_id"].is_unique
    # ~100 m target: mean should be well under 150 m (many edges are shorter).
    assert seg["length_m"].mean() < 150
    assert seg.crs.to_epsg() == 4326


@pytest.mark.skipif(not SEGMENTS.exists(), reason="segments not built (run features pipeline)")
def test_segments_endpoint_returns_geojson():
    from fastapi.testclient import TestClient

    from aquaroute.api.main import app

    client = TestClient(app)
    r = client.get("/segments", params={"classes": "primary,secondary", "limit": 50})
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) <= 50
    if fc["features"]:
        props = fc["features"][0]["properties"]
        assert "susceptibility" in props and "road_class" in props
        assert fc["features"][0]["geometry"]["type"] in ("LineString", "MultiLineString")
