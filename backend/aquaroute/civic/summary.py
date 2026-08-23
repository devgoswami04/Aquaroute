"""Civic decision-support summary (Module 8 backend, Phase 9).

Turns the pipeline's outputs into a planner-facing view: which roads flood most
often (the drainage-prioritisation evidence base), how predictions compare to
observations, and how the self-calibration loop is doing. This is the "civic
decision support, not a Maps clone" framing the brief insists on (§1).
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from aquaroute.config import get_config


def _labels_by_event() -> dict[str, pd.DataFrame]:
    from aquaroute.labels.store import list_labelled_events, load_labels
    return {s: load_labels(s) for s in list_labelled_events()}


def chronic_ranking(top_n: int = 20) -> dict:
    """Rank segments by how often they flood across the labelled events."""
    labels = _labels_by_event()
    n_events = len(labels)
    if n_events == 0:
        return {"n_events": 0, "segments": []}

    frames = []
    for ev, df in labels.items():
        d = df[["segment_id", "flooded", "depth_proxy"]].copy()
        d["event"] = ev
        frames.append(d)
    alll = pd.concat(frames, ignore_index=True)
    g = alll.groupby("segment_id").agg(
        events_flooded=("flooded", "sum"),
        mean_depth_proxy=("depth_proxy", lambda s: float(np.mean(s[s > 0])) if (s > 0).any() else 0.0),
    ).reset_index()
    g["n_events"] = n_events
    g["chronic_score"] = g["events_flooded"] / n_events
    g = g.sort_values(["chronic_score", "mean_depth_proxy"], ascending=False).head(top_n)

    # Attach class + centroid.
    from aquaroute.db.segments_store import load_segments_gdf
    seg = load_segments_gdf().set_index("segment_id")
    rows = []
    for _, r in g.iterrows():
        sid = r["segment_id"]
        cls = str(seg.loc[sid, "road_class"]) if sid in seg.index else None
        cen = seg.loc[sid, "geometry"].centroid if sid in seg.index else None
        rows.append({
            "segment_id": sid,
            "road_class": cls,
            "events_flooded": int(r["events_flooded"]),
            "n_events": n_events,
            "chronic_score": round(float(r["chronic_score"]), 3),
            "mean_depth_proxy": round(float(r["mean_depth_proxy"]), 3),
            "centroid": [round(cen.x, 5), round(cen.y, 5)] if cen is not None else None,
        })
    return {"n_events": n_events, "segments": rows}


def predicted_vs_observed() -> dict:
    """Compare FRF-derived flooded (peak > threshold) to observed labels per event."""
    try:
        from aquaroute.model.evaluate import classification_metrics
        from aquaroute.model.frf_targets import ONSET_THRESHOLD_M
        from aquaroute.model.predictor import get_predictor
    except Exception:
        return {"available": False}

    labels = _labels_by_event()
    if not labels:
        return {"available": False}
    try:
        pred = get_predictor()
    except Exception as e:
        return {"available": False, "note": str(e)}

    per_event = {}
    for ev, df in labels.items():
        peaks = pred.raw_peaks(ev)                      # {sid: peak}
        obs = df.set_index("segment_id")["flooded"].astype(int)
        common = [s for s in obs.index if s in peaks]
        y_true = obs.loc[common].to_numpy()
        y_prob = np.array([peaks[s] for s in common])
        y_pred = (y_prob > ONSET_THRESHOLD_M).astype(int)
        m = classification_metrics(y_true, y_pred, y_prob)
        per_event[ev] = {"f1": m["f1"], "precision": m["precision"],
                         "recall": m["recall"], "roc_auc": m.get("roc_auc")}
    mean_f1 = round(float(np.mean([v["f1"] for v in per_event.values()])), 3)
    return {"available": True, "per_event": per_event, "mean_f1": mean_f1}


def calibration_status() -> dict:
    from aquaroute.calibration.engine import get_engine
    s = get_engine().summary()
    n = s.get("segments_retired", 0)
    s["message"] = (f"Self-corrected {n} segment(s): flood predictions retired after "
                    f"observed repairs/de-silting." if n else
                    "No segments retired yet — feed observations via /ingest to calibrate.")
    return s


@functools.lru_cache(maxsize=1)
def _cached_summary(top_n: int) -> dict:
    cfg = get_config()
    from aquaroute.db.segments_store import load_segments_gdf
    n_segments = int(len(load_segments_gdf()))
    ranking = chronic_ranking(top_n)
    return {
        "corridor": {
            "place": cfg.place,
            "segments": n_segments,
            "events": [e["name"] for e in cfg.events],
        },
        "chronic": ranking,
        "predicted_vs_observed": predicted_vs_observed(),
        "calibration": calibration_status(),
    }


def build_civic_summary(top_n: int = 20, refresh: bool = False) -> dict:
    if refresh:
        _cached_summary.cache_clear()
    # calibration status is cheap & changes with cycles → always refresh that part
    out = _cached_summary(top_n)
    out["calibration"] = calibration_status()
    return out
