"""Synthetic real-time feeds (Module 1 adapters).

No free public API exists for water-level sensors, municipal public-works feeds,
or citizen reports on our corridor, so we generate them with an **identical JSON
contract** to a real MQTT/REST feed. The self-calibration loop (Module 5) consumes
these exactly as it would a real feed — swap the generator for a broker later and
nothing downstream changes (brief §4 note, §2.2).

Each function returns plain JSON-serialisable dicts.
"""
from __future__ import annotations

import numpy as np


def stream_sensor_readings(segment_ids, true_depths, ts: str, noise: float = 0.03,
                           seed: int = 0) -> list[dict]:
    """Water-level sensor readings: observed depth ≈ true depth + noise (m)."""
    rng = np.random.default_rng(seed)
    out = []
    for sid, d in zip(segment_ids, true_depths):
        obs = float(max(0.0, d + rng.normal(0, noise)))
        out.append({"segment_id": sid, "ts": ts, "depth_obs": round(obs, 3),
                    "source": "sensor"})
    return out


def stream_traffic_flow(segment_ids, true_depths, ts: str, freeflow_kmph: float = 40.0,
                        seed: int = 0) -> list[dict]:
    """Traffic speeds: water slows a road; ~free-flow speed ⇒ passable.

    Real flow APIs (TomTom/HERE) return current vs free-flow speed; we mimic that.
    """
    rng = np.random.default_rng(seed + 1)
    out = []
    for sid, d in zip(segment_ids, true_depths):
        slow = min(d / 0.30, 0.9)                     # depth 0.3 m ⇒ ~90% slowdown
        speed = float(max(1.0, freeflow_kmph * (1 - slow) + rng.normal(0, 1.5)))
        out.append({"segment_id": sid, "ts": ts, "speed_kmph": round(speed, 1),
                    "freeflow_kmph": freeflow_kmph, "source": "traffic"})
    return out


def poll_public_works(completed_ids, ts: str, work_type: str = "desilting") -> list[dict]:
    """Municipal public-works completions (e.g. a drain de-silted / road raised)."""
    return [{"segment_id": sid, "ts": ts, "status": "completed",
             "work_type": work_type, "source": "works"} for sid in completed_ids]


def submit_citizen_report(segment_id: str, status: str, ts: str,
                          depth_est: float | None = None, note: str = "") -> dict:
    """A single citizen flood/clear report (same shape as the /report endpoint)."""
    return {"segment_id": segment_id, "ts": ts, "status": status,
            "depth_est": depth_est, "note": note, "source": "report"}
