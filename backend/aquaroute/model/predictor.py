"""Live FRF prediction service (Module 4 → Module 7, Phase 6).

Loads the trained Flood Response Function once and runs it on the *current*
Open-Meteo rainfall forecast to produce a per-segment depth-vs-time curve and
onset/peak/clearance events. Results are cached in-process and served by the
/predict and /segment/{id}/curve endpoints.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from aquaroute.config import get_config
from aquaroute.model.frf_targets import ONSET_THRESHOLD_M

HORIZON = 24


def current_forecast_hyetograph(hours: int = HORIZON) -> np.ndarray:
    """bbox-mean hourly precip for the live forecast; bell-storm fallback offline."""
    try:
        from aquaroute.ingestion.rainfall import fetch_rainfall_forecast
        rain = fetch_rainfall_forecast(hours=hours)
        hourly = rain[rain["resolution"] == "hourly"]
        series = hourly.groupby("time")["precip_mm"].mean().sort_index().fillna(0.0)
        vals = series.to_numpy()[:hours]
        if len(vals) < hours:
            vals = np.pad(vals, (0, hours - len(vals)))
        return vals.astype("float32")
    except Exception:
        from aquaroute.model.frf_targets import _bell_hyetograph
        return _bell_hyetograph(hours).astype("float32")


def _vectorized_events(depth: np.ndarray, thresh: float = ONSET_THRESHOLD_M) -> pd.DataFrame:
    """Onset/peak/clearance (hours) + peak depth for every segment, vectorised."""
    N, T = depth.shape
    tgrid = np.arange(T)[None, :]
    above = depth > thresh
    any_flood = above.any(axis=1)
    onset = np.where(any_flood, above.argmax(axis=1), -1)
    peak_idx = depth.argmax(axis=1)
    peak_depth = depth.max(axis=1)
    below_after = (~above) & (tgrid >= peak_idx[:, None])
    clearance = np.where(below_after.any(axis=1), below_after.argmax(axis=1), T - 1)
    clearance = np.where(any_flood, clearance, -1)
    return pd.DataFrame({
        "onset": onset, "peak": peak_idx, "clearance": clearance,
        "peak_depth": np.round(peak_depth, 3),
    })


class Predictor:
    def __init__(self):
        self._loaded = False
        self.model = None
        self.stats = None
        self.node_x = None      # standardized [N, F]
        self.edge_index = None
        self.ids = None
        self.gdf = None         # segments (geometry) in graph order
        self.depth = None       # cached current-forecast curves [N, T]
        self.events = None      # cached events DataFrame
        self.hyeto = None

    def load(self):
        if self._loaded:
            return
        import torch

        from aquaroute.db.segments_store import load_segments_gdf
        from aquaroute.model.frf import FloodResponseFunction
        from aquaroute.model.graph import build_segment_edge_index

        ckpt_path = get_config().cache_dir / "models" / "frf.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                "FRF model not found. Train it: `python -m aquaroute.model.frf_pipeline`.")
        ck = torch.load(ckpt_path, weights_only=False)

        gdf = load_segments_gdf()
        edge_index, ids = build_segment_edge_index(gdf)
        gdf = gdf.set_index("segment_id").loc[ids].reset_index()
        feats = gdf.copy()
        feats["is_underpass"] = feats["is_underpass"].astype(int)

        raw = feats[ck["node_features"]].to_numpy(dtype="float64")
        med = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
        bad = np.where(~np.isfinite(raw))
        raw[bad] = np.take(med, bad[1])
        self.node_x = ((raw - ck["mean"]) / ck["std"]).astype("float32")

        model = FloodResponseFunction(len(ck["node_features"]), ck["horizon"],
                                      ck["hidden"], ck["gnn"], ck["temporal"])
        model.load_state_dict(ck["state_dict"])
        model.eval()

        self.model = model
        self.edge_index = edge_index
        self.ids = ids
        self.gdf = gdf
        self.id_to_idx = {sid: i for i, sid in enumerate(ids)}
        self._cache = {}  # scenario -> {depth, events, hyeto}
        self._loaded = True

    def _hyeto_for(self, scenario: str) -> np.ndarray:
        if scenario in (None, "", "live"):
            return current_forecast_hyetograph(HORIZON)
        # a historical event slug → replay that storm
        from aquaroute.config import get_config
        from aquaroute.model.frf_targets import get_event_hyetograph
        from aquaroute.labels.store import slug

        for e in get_config().events:
            if slug(e["name"]) == scenario:
                return get_event_hyetograph(e["start"], e["end"], HORIZON)
        # unknown scenario → fall back to live
        return current_forecast_hyetograph(HORIZON)

    def predict_current(self, scenario: str = "live", refresh: bool = False):
        self.load()
        if scenario in self._cache and not refresh:
            return self._cache[scenario]
        from aquaroute.model.frf import predict_all

        hyeto = self._hyeto_for(scenario)
        depth = predict_all(self.model, self.node_x, self.edge_index, hyeto)
        # Low-rainfall damping: the FRF is trained on 0–300 mm storms and is
        # uncalibrated at the extreme low end (a ~2 mm forecast should flood
        # nothing, per the reservoir physics). Scale depths by a smooth gate that
        # is 0 below ~2 mm and 1 by ~20 mm, so a dry live forecast reads as safe
        # while the heavy historical storms (>190 mm) are untouched.
        total = float(np.sum(hyeto))
        gate = float(np.clip((total - 2.0) / 18.0, 0.0, 1.0))
        depth = depth * gate
        entry = {"depth": depth, "events": _vectorized_events(depth),
                 "hyeto": hyeto, "total_mm": total, "gate": gate}
        self._cache[scenario] = entry
        return entry

    def predict_geojson(self, classes=None, bbox=None, limit=None,
                        only_flooded: bool = False, refresh: bool = False,
                        scenario: str = "live") -> dict:
        import json

        from shapely import set_precision

        entry = self.predict_current(scenario=scenario, refresh=refresh)
        # Apply the self-calibration correction (Module 5): retired segments read
        # shallower/dry. alpha=1 for segments the loop hasn't touched.
        from aquaroute.calibration.engine import get_engine
        alpha = get_engine().alpha_vector(self.ids)[:, None]
        cal_depth = entry["depth"] * alpha
        ev = _vectorized_events(cal_depth)
        gdf = self.gdf.copy()
        gdf["peak_depth"] = ev["peak_depth"].to_numpy()
        gdf["onset"] = ev["onset"].to_numpy()
        gdf["peak"] = ev["peak"].to_numpy()
        gdf["clearance"] = ev["clearance"].to_numpy()

        if classes:
            gdf = gdf[gdf["road_class"].astype(str).isin(classes)]
        if bbox:
            w, s, e, n = bbox
            gdf = gdf.cx[w:e, s:n]
        if only_flooded:
            gdf = gdf[gdf["peak_depth"] > ONSET_THRESHOLD_M]
        if limit is not None:
            gdf = gdf.iloc[:limit]

        gdf = gdf.copy()
        gdf["geometry"] = set_precision(gdf.geometry.values, grid_size=1e-5)
        keep = ["segment_id", "road_class", "peak_depth", "onset", "peak", "clearance", "geometry"]
        return json.loads(gdf[keep].to_json())

    def raw_peaks(self, scenario: str = "live") -> dict:
        """Uncalibrated FRF peak depth per segment — the 'prediction' the
        calibration cycle compares against observations."""
        entry = self.predict_current(scenario=scenario)
        peaks = entry["events"]["peak_depth"].to_numpy()
        return {sid: float(peaks[i]) for i, sid in enumerate(self.ids)}

    def segment_curve(self, segment_id: str, scenario: str = "live",
                      refresh: bool = False) -> dict:
        entry = self.predict_current(scenario=scenario, refresh=refresh)
        if segment_id not in self.id_to_idx:
            raise KeyError(segment_id)
        i = self.id_to_idx[segment_id]
        from aquaroute.calibration.engine import get_engine
        alpha = get_engine().alpha(segment_id)
        depth = entry["depth"][i] * alpha
        row = _vectorized_events(depth[None, :]).iloc[0]
        return {
            "segment_id": segment_id,
            "scenario": scenario,
            "calibration_alpha": round(float(alpha), 3),
            "t": list(range(len(depth))),
            "depth": [round(float(d), 3) for d in depth],
            "onset": None if row["onset"] < 0 else int(row["onset"]),
            "peak": int(row["peak"]),
            "clearance": None if row["clearance"] < 0 else int(row["clearance"]),
            "peak_depth": float(row["peak_depth"]),
            "hyetograph": [round(float(h), 2) for h in entry["hyeto"]],
        }


@functools.lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
