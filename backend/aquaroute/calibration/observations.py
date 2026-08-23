"""Turn heterogeneous feeds into per-segment flood observations (Module 5).

Sensors give depth directly. Traffic is inverted into a non-flood signal (Yuan et
al.'s slowdown indicator, flipped): near free-flow speed under heavy rain ⇒ the
road is passable ⇒ observed depth ≈ 0. Citizen reports map status→depth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def traffic_as_nonflood_signal(traffic_records, rain_intensity: float,
                               passable_ratio: float = 0.7,
                               rain_threshold: float = 2.0) -> list[dict]:
    """Invert traffic speed into flood observations.

    Only informative under meaningful rain (else a slow road is just congestion).
    speed/free-flow ≥ ``passable_ratio`` ⇒ passable (depth_obs 0); otherwise
    estimate depth from the slowdown (inverse of the synthetic traffic model).
    """
    if rain_intensity < rain_threshold:
        return []  # slowdowns aren't flood evidence without rain
    out = []
    for r in traffic_records:
        ratio = r["speed_kmph"] / max(r["freeflow_kmph"], 1e-6)
        if ratio >= passable_ratio:
            depth = 0.0
        else:
            depth = round(0.30 * (1.0 - ratio), 3)  # inverse of feeds.stream_traffic_flow
        out.append({"segment_id": r["segment_id"], "ts": r.get("ts"),
                    "depth_obs": depth, "source": "traffic",
                    "confidence": 0.6})
    return out


def _report_to_obs(r: dict) -> dict:
    flooded = str(r.get("status", "")).lower() == "flooded"
    depth = r.get("depth_est")
    if depth is None:
        depth = 0.3 if flooded else 0.0
    return {"segment_id": r["segment_id"], "ts": r.get("ts"),
            "depth_obs": float(depth), "source": "report", "confidence": 0.7}


def normalize_observations(records, rain_intensity: float = 5.0) -> pd.DataFrame:
    """Unify sensor/traffic/report feed records into one observation table.

    Returns columns [segment_id, depth_obs, source, confidence]. Multiple
    observations per segment are aggregated (confidence-weighted mean).
    """
    obs: list[dict] = []
    traffic = [r for r in records if r.get("source") == "traffic"]
    if traffic:
        obs.extend(traffic_as_nonflood_signal(traffic, rain_intensity))
    for r in records:
        src = r.get("source")
        if src == "sensor":
            obs.append({"segment_id": r["segment_id"], "ts": r.get("ts"),
                        "depth_obs": float(r["depth_obs"]), "source": "sensor",
                        "confidence": 0.9})
        elif src == "report":
            obs.append(_report_to_obs(r))
        # traffic already handled; works events carry no depth
    if not obs:
        return pd.DataFrame(columns=["segment_id", "depth_obs", "source", "confidence"])

    df = pd.DataFrame(obs)
    agg = (df.assign(wd=df["depth_obs"] * df["confidence"])
             .groupby("segment_id")
             .apply(lambda g: pd.Series({
                 "depth_obs": g["wd"].sum() / g["confidence"].sum(),
                 "source": ",".join(sorted(set(g["source"]))),
                 "confidence": g["confidence"].max(),
             }), include_groups=False)
             .reset_index())
    return agg
