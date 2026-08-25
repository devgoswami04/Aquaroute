"""Depth-vs-time training targets for the Flood Response Function (Phase 5).

SAR gives only snapshots, so we synthesise physically-motivated depth-vs-time
curves as targets: a per-segment **linear reservoir** driven by a rainfall
hyetograph. Water rises during rain and recedes afterward at a rate set by terrain
(depressions/flat ground drain slower); the peak magnitude scales with storm
severity and the segment's susceptibility.

Crucially the target depends on the *storm* and *terrain* only — not on a discrete
flood label — so we can generate unlimited (storm → curve) training pairs with
varied onset/duration/intensity. That is what lets the temporal encoder actually
learn the rainfall-shape → timing mapping and generalise it to the 3 real
historical storms (which are never used in training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ONSET_THRESHOLD_M = 0.10   # depth (m) that counts as flooded / impassable onset
MAX_DEPTH_M = 0.90         # saturation depth for a very severe storm on worst terrain
S_REF = 25.0               # reservoir storage knee (controls severity → depth)


def _bell_hyetograph(T: int, total: float = 40.0, peak_frac: float = 0.4,
                     width_frac: float = 0.15) -> np.ndarray:
    t = np.arange(T)
    peak = T * peak_frac
    r = np.exp(-((t - peak) ** 2) / (2 * (T * width_frac) ** 2))
    return (r / r.sum()) * total


def random_storm(rng: np.random.Generator, T: int = 24) -> np.ndarray:
    """A random but plausible storm hyetograph: varied onset, width, and total.

    Totals span 0–300 mm (including near-dry days) so the model is calibrated at
    the low end and predicts ~no flooding on a light live forecast, not just on the
    heavy training storms.
    """
    total = float(rng.uniform(0.0, 300.0))       # event total (mm), incl. dry
    peak_frac = float(rng.uniform(0.15, 0.7))    # when the burst peaks
    width_frac = float(rng.uniform(0.06, 0.22))  # burst duration
    base = _bell_hyetograph(T, total, peak_frac, width_frac)
    base = base + rng.uniform(0, 0.3, size=T)     # light background drizzle
    return base.astype("float32")


def get_event_hyetograph(start: str, end: str, T: int = 24) -> np.ndarray:
    """bbox-mean hourly precip over the wettest T-hour window of a real event."""
    try:
        from aquaroute.ingestion.rainfall import fetch_rainfall_history
        rain = fetch_rainfall_history(start=start, end=end)
        hourly = rain[rain["resolution"] == "hourly"]
        series = hourly.groupby("time")["precip_mm"].mean().sort_index().fillna(0.0)
        vals = series.to_numpy()
        if len(vals) < T:
            vals = np.pad(vals, (0, T - len(vals)))
        best_s, best_i = -1.0, 0
        for i in range(0, len(vals) - T + 1):
            s = vals[i:i + T].sum()
            if s > best_s:
                best_s, best_i = s, i
        window = vals[best_i:best_i + T]
        return (window if window.sum() > 0 else _bell_hyetograph(T)).astype("float32")
    except Exception:
        return _bell_hyetograph(T).astype("float32")


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    lo, hi = np.nanpercentile(a[finite], 2), np.nanpercentile(a[finite], 98)
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.nan_to_num(np.clip((a - lo) / (hi - lo), 0, 1))


def segment_reservoir_params(feats: pd.DataFrame, propensity=None) -> dict:
    """Static per-segment reservoir coefficients derived from terrain/surface.

    ``propensity`` (per-segment flood probability learned from real SAR) grounds
    the peak-depth gate in observed flooding when supplied; otherwise the gate
    falls back to a terrain susceptibility heuristic.
    """
    imperv = np.nan_to_num(feats["imperviousness"].fillna(0.6).to_numpy(), nan=0.6)
    up = _norm(feats["upstream_area"].to_numpy())
    depr = _norm(feats["depression_depth"].to_numpy())
    slope = _norm(feats["slope"].to_numpy())

    if propensity is not None:
        # Real-SAR-grounded gate: flood extent matches observed (calibrated prob).
        gate = 0.9 * np.clip(np.asarray(propensity, dtype="float64"), 0.0, 1.0)
    else:
        sus = _norm(feats["susceptibility"].to_numpy()) if "susceptibility" in feats else \
            np.clip(0.4 * _norm(feats["twi"].to_numpy()) + 0.6 * depr, 0, 1)
        gate = 0.9 * (sus ** 1.3)
    return {
        "coeff": 0.3 + 0.7 * imperv,                 # runoff coefficient
        "upboost": 1.0 + 0.5 * up,                    # upstream contribution
        "k": np.clip(0.5 - 0.35 * depr - 0.10 * (1 - slope), 0.08, 0.5),  # recession/h
        "sus_gate": gate,                             # peak-depth gate (0..0.9)
    }


def depth_from_storm(params: dict, hyeto: np.ndarray) -> np.ndarray:
    """[N, T] depth curves for a storm — timing from dynamics, magnitude from
    severity × susceptibility. Absolute (label-independent)."""
    T = len(hyeto)
    coeff, upboost, k, gate = (params["coeff"], params["upboost"],
                               params["k"], params["sus_gate"])
    N = len(coeff)
    S = np.zeros(N)
    depth = np.zeros((N, T), dtype="float32")
    for t in range(T):
        S = np.maximum(S + coeff * upboost * float(hyeto[t]) - k * S, 0.0)
        depth[:, t] = MAX_DEPTH_M * gate * np.tanh(S / S_REF)
    return np.nan_to_num(depth)


def storm_shape(params: dict, hyeto: np.ndarray) -> np.ndarray:
    """[N, T] **normalised temporal shape** (peak 1 per segment) of the flood
    response — the *timing* only, independent of amplitude/extent.

    This is what the Flood Response Function learns: dense (every rained-on
    segment has a shape), so training is stable. The flood *extent/magnitude* is
    supplied separately by the real-SAR flood propensity at inference:
        depth = MAX_DEPTH_M · propensity · shape
    """
    T = len(hyeto)
    coeff, upboost, k = params["coeff"], params["upboost"], params["k"]
    N = len(coeff)
    S = np.zeros(N)
    resp = np.zeros((N, T), dtype="float64")
    for t in range(T):
        S = np.maximum(S + coeff * upboost * float(hyeto[t]) - k * S, 0.0)
        resp[:, t] = np.tanh(S / S_REF)
    peak = resp.max(axis=1, keepdims=True)
    shape = np.divide(resp, peak, out=np.zeros_like(resp), where=peak > 1e-6)
    return np.nan_to_num(shape).astype("float32")


def synthesize_depth_curves(feats: pd.DataFrame, hyeto: np.ndarray) -> np.ndarray:
    """Convenience: reservoir params + storm → depth curves ([N, T])."""
    return depth_from_storm(segment_reservoir_params(feats), hyeto)


def derive_events_from_curve(depth: np.ndarray, thresh: float = ONSET_THRESHOLD_M) -> dict:
    """Onset/peak/clearance (hours) + peak depth from a single depth curve."""
    depth = np.asarray(depth)
    peak_idx = int(np.argmax(depth))
    peak_depth = float(depth[peak_idx])
    above = np.where(depth > thresh)[0]
    if above.size == 0:
        return {"onset": None, "peak": peak_idx, "clearance": None, "peak_depth": peak_depth}
    onset = int(above[0])
    after = [t for t in range(peak_idx, len(depth)) if depth[t] <= thresh]
    clearance = int(after[0]) if after else int(len(depth) - 1)
    return {"onset": onset, "peak": peak_idx, "clearance": clearance, "peak_depth": peak_depth}
