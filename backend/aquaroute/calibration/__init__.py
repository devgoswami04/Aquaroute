"""Module 5 — closed-loop self-calibration (the review fix).

Per-segment residuals + change-point detection (ruptures) + online update (river-
style EWMA) so the model reacts to structural change and retires stale flood
predictions automatically — e.g. after a road is de-silted or repaired.
"""
from aquaroute.calibration.changepoint import detect_change_point
from aquaroute.calibration.engine import CalibrationEngine, get_engine
from aquaroute.calibration.observations import (
    normalize_observations,
    traffic_as_nonflood_signal,
)
from aquaroute.calibration.residuals import compute_residuals

__all__ = [
    "compute_residuals",
    "detect_change_point",
    "traffic_as_nonflood_signal",
    "normalize_observations",
    "CalibrationEngine",
    "get_engine",
]
