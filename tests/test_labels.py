"""Phase 3 tests — labelling logic, merge, and the /labels endpoint.

Pure-logic tests run always; the endpoint/training tests skip if the Phase 3
label artifacts aren't present.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO / "data" / "labels"
HAVE_LABELS = LABELS_DIR.exists() and any(LABELS_DIR.glob("*.parquet"))


def test_merge_report_labels_overrides_sar():
    from aquaroute.labels.merge import merge_report_labels

    sar = pd.DataFrame({
        "segment_id": ["a", "b", "c"],
        "flooded": [True, False, True],
        "depth_proxy": [0.4, 0.0, 0.3],
        "source": ["synthetic"] * 3,
    })
    reports = pd.DataFrame({
        "segment_id": ["a", "b"],
        "status": ["clear", "flooded"],
        "depth_est": [0.0, 0.5],
        "note": ["", ""],
    })
    out = merge_report_labels(sar, reports).set_index("segment_id")
    assert out.loc["a", "flooded"] == False and out.loc["a", "source"] == "report"
    assert out.loc["b", "flooded"] == True and out.loc["b", "depth_proxy"] == 0.5
    assert out.loc["c", "flooded"] == True and out.loc["c", "source"] == "synthetic"


def test_severity_monotonic_in_rainfall():
    from aquaroute.synthetic.flood_labels import severity_from_rainfall

    assert severity_from_rainfall(50) < severity_from_rainfall(300)
    assert 0.05 <= severity_from_rainfall(0) <= severity_from_rainfall(10_000) <= 0.40


@pytest.mark.skipif(not HAVE_LABELS, reason="labels not built (run labels pipeline)")
def test_labels_endpoint_overlay():
    from fastapi.testclient import TestClient

    from aquaroute.api.main import app

    client = TestClient(app)
    evs = client.get("/events").json()
    labelled = [e["name"] for e in evs if e["labelled"]]
    assert labelled, "expected at least one labelled event"

    r = client.get("/labels", params={"event": labelled[0], "classes": "primary,secondary", "only_flooded": True})
    assert r.status_code == 200
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    for feat in fc["features"]:
        assert feat["properties"]["flooded"] is True
        assert feat["properties"]["depth_proxy"] >= 0


@pytest.mark.skipif(not HAVE_LABELS, reason="labels not built")
def test_training_set_shapes():
    from aquaroute.labels.training_set import assemble_training_set

    X, y = assemble_training_set()
    assert len(X) == len(y) and len(X) > 0
    assert {"flooded", "depth_proxy"}.issubset(y.columns)
    assert set(y["flooded"].unique()).issubset({0, 1})
