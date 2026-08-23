"""Phase 8 tests — vehicle-aware routing.

The search logic is tested on a tiny hand-built graph (no model needed): the
direct edge floods, and only a vehicle with a high enough threshold may take it.
An end-to-end test via /route is guarded on the FRF model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FRF_CKPT = REPO / "data" / "models" / "frf.pt"


def test_vehicle_thresholds_ordered():
    from aquaroute.routing.vehicle import get_threshold, vehicle_thresholds

    t = vehicle_thresholds()
    assert t["two_wheeler"] < t["auto"] < t["car"] < t["bus"]
    assert get_threshold("bus") == 0.60
    with pytest.raises(ValueError):
        get_threshold("submarine")


def _tiny_router():
    import networkx as nx

    from aquaroute.routing.router import RouterContext

    G = nx.DiGraph()

    def dc(v):
        return np.full(24, v, dtype="float32")

    # Direct edge is short but floods to 0.40 m; the dry detour is longer, so a bus
    # (tolerates 0.40 m) rationally takes the direct edge while a two-wheeler must
    # detour — different routes on the same graph.
    G.add_edge(0, 3, length=100, base_time=2.0, depth=dc(0.40))   # direct, floods
    G.add_edge(0, 1, length=200, base_time=3.0, depth=dc(0.0))    # detour, dry, longer
    G.add_edge(1, 2, length=200, base_time=3.0, depth=dc(0.0))
    G.add_edge(2, 3, length=200, base_time=3.0, depth=dc(0.0))

    R = RouterContext.__new__(RouterContext)
    R.G = G
    R.scenario = "test"
    R.node_xy = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}
    R._nodes = np.array([0, 1, 2, 3])
    R._coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    return R


def test_search_is_vehicle_aware():
    from aquaroute.routing.router import RISK_WEIGHT

    R = _tiny_router()
    # two-wheeler (0.15 m) can't cross the 0.40 m direct edge → must detour
    p_2w = R._time_dependent_dijkstra(0, 3, 0, 0.15, RISK_WEIGHT, True)
    assert p_2w == [0, 1, 2, 3]
    # bus (0.60 m) tolerates 0.40 m → takes the short direct edge
    p_bus = R._time_dependent_dijkstra(0, 3, 0, 0.60, RISK_WEIGHT, True)
    assert p_bus == [0, 3]


def test_evaluate_reports_blocked_metres():
    R = _tiny_router()
    ev = R._evaluate([0, 3], depart_hour=0, threshold=0.15)   # crossing the flooded edge
    assert ev["blocked_m"] == 100.0 and ev["max_depth_m"] == 0.4


@pytest.mark.skipif(not FRF_CKPT.exists(), reason="FRF model not trained")
def test_route_endpoint_vehicle_classes_differ():
    from fastapi.testclient import TestClient

    from aquaroute.api.main import app

    client = TestClient(app)
    body = {"origin": [80.135, 12.95], "dest": [80.23, 12.945],
            "depart_time": 18, "scenario": "2021_chennai_floods"}
    r2w = client.post("/route", json={**body, "vehicle": "two_wheeler"}).json()
    rbus = client.post("/route", json={**body, "vehicle": "bus"}).json()
    assert r2w["safe_route"]["geometry"] != rbus["safe_route"]["geometry"]
    # the stricter vehicle should not have a shorter safe route
    assert r2w["safe_route"]["distance_m"] >= rbus["safe_route"]["distance_m"]
