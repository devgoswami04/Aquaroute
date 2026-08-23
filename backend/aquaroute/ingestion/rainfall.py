"""Rainfall ingestion from Open-Meteo (Module 1).

Open-Meteo is keyless and free (CC-BY). It serves *point* queries, so to cover a
bbox we sample a coarse grid of points and request them in a single call
(the API accepts comma-separated latitude/longitude lists). The result is a tidy
long DataFrame — one row per (grid point, timestamp) — which is what the feature
layer (Module 2) and PostGIS `rainfall` table (brief §7) expect.

Public functions
----------------
fetch_rainfall_forecast(bbox, hours=24) -> DataFrame
fetch_rainfall_history(bbox, start, end)  -> DataFrame
"""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd
import requests

from aquaroute.config import BBox, get_config

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_TIMEOUT = 60


def _grid_points(bbox: BBox, n: int = 3) -> list[tuple[float, float]]:
    """An n x n lattice of (lat, lon) sample points across the bbox (inclusive)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [bbox.center()]
    lats = [bbox.south + (bbox.north - bbox.south) * i / (n - 1) for i in range(n)]
    lons = [bbox.west + (bbox.east - bbox.west) * j / (n - 1) for j in range(n)]
    return [(round(la, 4), round(lo, 4)) for la in lats for lo in lons]


def _melt_open_meteo(payload, points, value_key: str, block: str) -> pd.DataFrame:
    """Flatten Open-Meteo's per-location response block into long rows.

    Open-Meteo returns a single dict when one location is queried and a list of
    dicts when several are. `block` is 'hourly' or 'minutely_15'.
    """
    locations = payload if isinstance(payload, list) else [payload]
    frames: list[pd.DataFrame] = []
    for (lat, lon), loc in zip(points, locations):
        b = loc.get(block)
        if not b or "time" not in b:
            continue
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(b["time"]),
                "precip_mm": b.get(value_key, [None] * len(b["time"])),
            }
        )
        df["lat"] = lat
        df["lon"] = lon
        df["point_id"] = f"{lat:.4f},{lon:.4f}"
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["point_id", "lat", "lon", "time", "precip_mm"])
    out = pd.concat(frames, ignore_index=True)
    return out[["point_id", "lat", "lon", "time", "precip_mm"]]


def _request(url: str, points: Iterable[tuple[float, float]], **params) -> dict | list:
    pts = list(points)
    params["latitude"] = ",".join(str(la) for la, _ in pts)
    params["longitude"] = ",".join(str(lo) for _, lo in pts)
    params["timezone"] = "auto"
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_rainfall_forecast(bbox: BBox | None = None, hours: int = 24,
                            grid_n: int = 3) -> pd.DataFrame:
    """Hourly + 15-min precipitation forecast over the bbox.

    Returns a long DataFrame with columns
    ``[point_id, lat, lon, time, precip_mm, resolution]`` where ``resolution`` is
    either ``"hourly"`` or ``"minutely_15"``.
    """
    cfg = get_config()
    bbox = bbox or cfg.bbox
    points = _grid_points(bbox, grid_n)
    forecast_days = max(1, math.ceil(hours / 24))

    payload = _request(
        FORECAST_URL,
        points,
        hourly="precipitation",
        minutely_15="precipitation",
        forecast_days=min(forecast_days, 16),
    )
    hourly = _melt_open_meteo(payload, points, "precipitation", "hourly")
    hourly["resolution"] = "hourly"
    q15 = _melt_open_meteo(payload, points, "precipitation", "minutely_15")
    q15["resolution"] = "minutely_15"

    # Trim hourly to the requested horizon (relative to the first timestamp).
    if not hourly.empty:
        start = hourly["time"].min()
        hourly = hourly[hourly["time"] < start + pd.Timedelta(hours=hours)]
    return pd.concat([hourly, q15], ignore_index=True)


def fetch_rainfall_history(bbox: BBox | None = None, start: str = "", end: str = "",
                           grid_n: int = 3) -> pd.DataFrame:
    """Hourly historical precipitation (Open-Meteo Archive/ERA5) for a date range.

    `start`/`end` are ISO dates (YYYY-MM-DD). Same schema as the forecast API, so
    training events (brief §1) share the forecast code paths.
    """
    if not start or not end:
        raise ValueError("start and end (YYYY-MM-DD) are required for history")
    cfg = get_config()
    bbox = bbox or cfg.bbox
    points = _grid_points(bbox, grid_n)

    payload = _request(
        ARCHIVE_URL,
        points,
        hourly="precipitation",
        start_date=start,
        end_date=end,
    )
    hourly = _melt_open_meteo(payload, points, "precipitation", "hourly")
    hourly["resolution"] = "hourly"
    return hourly
