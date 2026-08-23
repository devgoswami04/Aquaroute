"""Phase 6 tests — live prediction service + endpoints.

The vectorised-events test always runs. The endpoint test skips if the FRF model
isn't trained; it works offline because the hyetograph fetch falls back to a
synthetic bell storm when the network is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FRF_CKPT = REPO / "data" / "models" / "frf.pt"


def test_vectorized_events():
    from aquaroute.model.predictor import _vectorized_events

    # seg0: dry; seg1: rises to 0.5 at t=3, clears by t=6
    depth = np.array([
        [0.0, 0.02, 0.03, 0.05, 0.04, 0.0, 0.0],
        [0.0, 0.05, 0.2, 0.5, 0.3, 0.08, 0.0],
    ])
    ev = _vectorized_events(depth, thresh=0.1)
    assert ev.loc[0, "onset"] == -1 and ev.loc[0, "clearance"] == -1
    assert ev.loc[1, "onset"] == 2 and ev.loc[1, "peak"] == 3
    assert ev.loc[1, "clearance"] == 5
    assert abs(ev.loc[1, "peak_depth"] - 0.5) < 1e-6


def _have_torch():
    try:
        import torch  # noqa
        import torch_geometric  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(not FRF_CKPT.exists() or not _have_torch(),
                    reason="FRF model not trained / torch missing")
def test_predict_and_curve_endpoints():
    from fastapi.testclient import TestClient

    from aquaroute.api.main import app

    client = TestClient(app)
    r = client.get("/predict", params={"classes": "primary,secondary", "scenario": "live"})
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection" and fc["features"]
    props = fc["features"][0]["properties"]
    assert "peak_depth" in props and "onset" in props and "clearance" in props

    sid = fc["features"][0]["properties"]["segment_id"]
    c = client.get(f"/segment/{sid}/curve")
    assert c.status_code == 200
    curve = c.json()
    assert len(curve["depth"]) == 24 and len(curve["hyetograph"]) == 24
    assert curve["peak"] is not None
