"""Phase 9 tests — civic decision-support summary."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LABELS_DIR = REPO / "data" / "labels"
FRF_CKPT = REPO / "data" / "models" / "frf.pt"
HAVE_LABELS = LABELS_DIR.exists() and any(LABELS_DIR.glob("*.parquet"))


def test_calibration_status_message():
    from aquaroute.civic.summary import calibration_status

    s = calibration_status()
    assert "segments_tracked" in s and "message" in s


@pytest.mark.skipif(not HAVE_LABELS, reason="labels not built")
def test_chronic_ranking_sorted_and_bounded():
    from aquaroute.civic.summary import chronic_ranking

    r = chronic_ranking(top_n=10)
    assert r["n_events"] >= 1
    segs = r["segments"]
    assert 0 < len(segs) <= 10
    scores = [s["chronic_score"] for s in segs]
    assert scores == sorted(scores, reverse=True)      # ranked
    assert all(0 <= s["chronic_score"] <= 1 for s in segs)
    assert all(s["centroid"] and len(s["centroid"]) == 2 for s in segs)


@pytest.mark.skipif(not (HAVE_LABELS and FRF_CKPT.exists()), reason="labels/FRF missing")
def test_civic_summary_endpoint():
    from fastapi.testclient import TestClient

    from aquaroute.api.main import app

    client = TestClient(app)
    d = client.get("/civic/summary", params={"top_n": 5}).json()
    assert d["corridor"]["segments"] > 1000
    assert len(d["chronic"]["segments"]) <= 5
    assert "message" in d["calibration"]
    pvo = d["predicted_vs_observed"]
    assert pvo["available"] and 0 <= pvo["mean_f1"] <= 1
