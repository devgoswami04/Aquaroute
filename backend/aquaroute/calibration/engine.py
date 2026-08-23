"""Closed-loop self-calibration engine (Module 5) — the review fix.

Keeps a per-segment multiplicative correction ``alpha`` on top of the (frozen)
Flood Response Function. Each calibration cycle compares predicted vs observed
depth, updates ``alpha`` online toward the observed regime, and — crucially —
reacts to **structural change** (ruptures change-point, or a completed public-works
event) by boosting the learning rate so stale flood predictions **retire within a
few events**, automatically. The FRF is never retrained on all data (brief §11);
correction is per-segment, residual- and change-point-driven.

corrected_depth_i = alpha_i · frf_depth_i
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aquaroute.calibration.changepoint import detect_change_point
from aquaroute.calibration.observations import normalize_observations
from aquaroute.calibration.residuals import compute_residuals
from aquaroute.config import get_config

BASE_LR = 0.35            # normal online learning rate (EWMA toward observed)
BOOST_LR = 0.85           # after a change-point / public-works reset
BOOST_EVENTS = 2          # how many cycles the boost persists
EPS = 1e-3
RETIRE_THRESHOLD = 0.10   # corrected peak below this ⇒ prediction retired (m)


class CalibrationEngine:
    """Per-segment online correction state, with disk persistence."""

    def __init__(self, path: Path | None = None):
        self.path = path or (get_config().cache_dir / "calibration_state.json")
        self.state: dict[str, dict] = {}
        self._obs_buffer: list[dict] = []   # feed records awaiting a cycle
        self.load()

    # --- persistence ---
    def load(self):
        if self.path.exists():
            self.state = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.state))

    def reset_all(self):
        self.state = {}
        self._obs_buffer = []
        self.save()

    # --- state helpers ---
    def _seg(self, sid: str) -> dict:
        return self.state.setdefault(
            sid, {"alpha": 1.0, "history": [], "boost": 0, "obs_count": 0})

    def alpha(self, sid: str) -> float:
        return self.state.get(sid, {}).get("alpha", 1.0)

    def alpha_vector(self, ids) -> np.ndarray:
        return np.array([self.state.get(sid, {}).get("alpha", 1.0) for sid in ids],
                        dtype="float32")

    # --- feed intake ---
    def ingest(self, records: list[dict]):
        """Buffer raw feed records (sensor/traffic/report) for the next cycle."""
        self._obs_buffer.extend(records)

    def apply_public_works_reset(self, segment_id: str, work_type: str = ""):
        """A completed works event ⇒ discount history + boost the learning rate so
        this segment adapts to its new (repaired) regime immediately."""
        st = self._seg(segment_id)
        st["boost"] = BOOST_EVENTS
        st["history"] = []       # old regime no longer representative
        return {"segment_id": segment_id, "action": "works_reset", "work_type": work_type}

    # --- the cycle ---
    def run_cycle(self, predictions, observations=None, works_ids=(),
                  rain_intensity: float = 5.0) -> pd.DataFrame:
        """One calibration cycle. ``predictions`` = dict/df of per-segment FRF peak
        depth; ``observations`` = normalized obs df (or None to use the buffer)."""
        for sid in works_ids:
            self.apply_public_works_reset(sid)

        if observations is None:
            observations = normalize_observations(self._obs_buffer, rain_intensity)
            self._obs_buffer = []
        if observations.empty:
            return pd.DataFrame(columns=["segment_id", "residual", "change_point",
                                         "action", "alpha", "pred", "obs"])

        res = compute_residuals(predictions, observations)
        log = []
        for _, row in res.iterrows():
            sid, pred, obs, residual = row["segment_id"], row["pred"], row["obs"], row["residual"]
            st = self._seg(sid)
            st["history"].append(float(residual))
            st["obs_count"] += 1

            cp = detect_change_point(st["history"])
            boosted = st["boost"] > 0 or cp
            lr = BOOST_LR if boosted else BASE_LR

            target = float(np.clip(obs / max(pred, EPS), 0.0, 2.0))
            st["alpha"] = float((1 - lr) * st["alpha"] + lr * target)
            if st["boost"] > 0:
                st["boost"] -= 1

            action = ("works_reset" if sid in works_ids
                      else "change_point" if cp else "update")
            log.append({"segment_id": sid, "residual": round(float(residual), 3),
                        "change_point": bool(cp), "action": action,
                        "alpha": round(st["alpha"], 3),
                        "pred": round(float(pred), 3), "obs": round(float(obs), 3)})
        self.save()
        return pd.DataFrame(log)

    def summary(self) -> dict:
        alphas = {s: v["alpha"] for s, v in self.state.items()}
        retired = [s for s, v in self.state.items()
                   if v["alpha"] < 0.5 and v["obs_count"] > 0]
        return {
            "segments_tracked": len(self.state),
            "segments_retired": len(retired),
            "retired_examples": retired[:10],
            "mean_alpha": round(float(np.mean(list(alphas.values()))), 3) if alphas else 1.0,
        }


@functools.lru_cache(maxsize=1)
def get_engine() -> CalibrationEngine:
    return CalibrationEngine()
