"""Phase 7 tests — self-calibration loop.

Pure-logic tests (residuals, traffic signal, online update, change-point) always
run. The end-to-end road-fixed test is guarded on the FRF model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FRF_CKPT = REPO / "data" / "models" / "frf.pt"


def test_compute_residuals():
    from aquaroute.calibration.residuals import compute_residuals

    preds = {"a": 0.5, "b": 0.2, "c": 0.4}
    obs = pd.DataFrame({"segment_id": ["a", "b"], "depth_obs": [0.5, 0.0]})
    r = compute_residuals(preds, obs).set_index("segment_id")
    assert abs(r.loc["a", "residual"] - 0.0) < 1e-9   # matches → no residual
    assert abs(r.loc["b", "residual"] - 0.2) < 1e-9   # predicted wet, observed dry
    assert "c" not in r.index                          # unobserved → excluded


def test_traffic_nonflood_signal():
    from aquaroute.calibration.observations import traffic_as_nonflood_signal

    traffic = [
        {"segment_id": "fast", "speed_kmph": 38, "freeflow_kmph": 40},
        {"segment_id": "jam", "speed_kmph": 6, "freeflow_kmph": 40},
    ]
    obs = {o["segment_id"]: o["depth_obs"] for o in traffic_as_nonflood_signal(traffic, rain_intensity=10)}
    assert obs["fast"] == 0.0            # near free-flow under rain ⇒ passable
    assert obs["jam"] > 0.1              # heavy slowdown ⇒ flooded
    # no rain ⇒ slowdowns aren't flood evidence
    assert traffic_as_nonflood_signal(traffic, rain_intensity=0) == []


def test_online_update_retires_repaired_segment():
    """A segment that stops flooding (obs→0) should have alpha decay below 0.5."""
    from aquaroute.calibration.engine import CalibrationEngine

    eng = CalibrationEngine(path=REPO / "data" / "_test_cal_state.json")
    eng.reset_all()
    preds = {"seg": 0.6}
    wet = pd.DataFrame({"segment_id": ["seg"], "depth_obs": [0.6]})
    dry = pd.DataFrame({"segment_id": ["seg"], "depth_obs": [0.0]})
    for _ in range(3):
        eng.run_cycle(preds, observations=wet)
    assert eng.alpha("seg") > 0.7                      # stays flooded while wet
    eng.apply_public_works_reset("seg")
    for _ in range(3):
        eng.run_cycle(preds, observations=dry)
    assert eng.alpha("seg") < 0.3                      # retired after repair
    eng.reset_all()


def test_change_point_detects_shift():
    from aquaroute.calibration.changepoint import detect_change_point

    assert not detect_change_point([0.0, 0.0, 0.0])                 # flat, too short
    assert detect_change_point([0.0, 0.0, 0.0, 0.5, 0.5, 0.5])      # clear regime shift


@pytest.mark.skipif(not FRF_CKPT.exists(), reason="FRF model not trained")
def test_road_fixed_end_to_end():
    from aquaroute.calibration.roadfixed import run_road_fixed_test

    r = run_road_fixed_test(n_events=8, repair_at=3)
    assert r["retired_works_after"] is not None      # works-fix retires
    assert r["retired_silent_after"] is not None      # silent-fix retires
    assert r["control_depth"] >= 0.1                   # control stays flooded
