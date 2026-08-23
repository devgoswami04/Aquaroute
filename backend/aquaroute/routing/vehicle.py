"""Vehicle classes and their flood depth thresholds (Module 6).

A 15 cm road stops a two-wheeler but not a bus — routing is class-aware. Thresholds
are the water depth (metres) at which a vehicle can no longer safely pass.
"""
from __future__ import annotations

from aquaroute.config import get_config

# Fallback if config.yaml lacks the block (brief §6 defaults).
_DEFAULTS = {"two_wheeler": 0.15, "auto": 0.25, "car": 0.30, "bus": 0.60}


def vehicle_thresholds() -> dict[str, float]:
    cfg = get_config().vehicle_thresholds
    return {**_DEFAULTS, **(cfg or {})}


def get_threshold(vehicle: str) -> float:
    t = vehicle_thresholds()
    if vehicle not in t:
        raise ValueError(f"unknown vehicle '{vehicle}'. options: {sorted(t)}")
    return t[vehicle]
