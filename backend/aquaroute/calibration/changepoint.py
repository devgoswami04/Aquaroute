"""Structural change-point detection on a segment's residual history (Module 5).

Uses `ruptures` (batch) to flag when a segment's predicted-vs-observed residual
series shifts regime — e.g. a chronically-flooded road that stops flooding after
de-silting. This is what makes the loop react to *structural* change rather than
just averaging everything (brief §11: not "retrain on all data").
"""
from __future__ import annotations

import numpy as np


def detect_change_point(series, pen: float = 0.15, min_size: int = 2) -> bool:
    """True if a change-point is detected in the residual series.

    Needs enough points to see a shift; returns False for short histories. The
    default penalty is tuned for depth-scale residuals (metres): a ~0.3 m regime
    shift is detected while sensor-noise-only series stay flat (no false alarms).
    """
    x = np.asarray(series, dtype="float64")
    if len(x) < 2 * min_size + 1:
        return False
    import ruptures as rpt

    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(x)
    bkps = algo.predict(pen=pen)          # includes len(x) as the final index
    interior = [b for b in bkps if 0 < b < len(x)]
    return len(interior) > 0


def river_drift_detector():
    """A streaming drift detector (river ADWIN) for online per-segment monitoring.

    Complements the batch `ruptures` pass; used in the road-fixed demo to confirm
    the shift is detectable online, one observation at a time.
    """
    from river import drift
    return drift.ADWIN()
