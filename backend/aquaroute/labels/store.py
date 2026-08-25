"""Persist & read per-event flood labels.

Parquet is the always-available cache (``data/labels/<event>.parquet``); PostGIS
``flood_labels`` is written too when the DB is up. The map overlay joins labels to
segment geometry from the segments store.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from aquaroute.config import get_config
from aquaroute.db.session import get_engine, postgis_available


def slug(event: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", event.lower()).strip("_")


def _labels_dir() -> Path:
    d = get_config().cache_dir / "labels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(event: str) -> Path:
    return _labels_dir() / f"{slug(event)}.parquet"


def save_labels(event: str, df: pd.DataFrame) -> dict:
    out = df.copy()
    out["event"] = event
    path = _path(event)
    out.to_parquet(path, index=False)

    wrote_pg = False
    if postgis_available():
        try:
            out.to_sql("flood_labels", get_engine(), if_exists="append", index=False)
            wrote_pg = True
        except Exception:
            wrote_pg = False
    return {
        "event": event,
        "count": int(len(out)),
        "flooded": int(out["flooded"].sum()),
        "postgis": wrote_pg,
        "file": str(path),
    }


def load_labels(event: str) -> pd.DataFrame:
    path = _path(event)
    if not path.exists():
        raise FileNotFoundError(f"No labels for '{event}'. Run the Phase 3 labels pipeline.")
    return pd.read_parquet(path)


def list_labelled_events() -> list[str]:
    return sorted(p.stem for p in _labels_dir().glob("*.parquet"))


def dominant_source(event: str) -> str:
    """The event's label provenance: 'sar' (real Sentinel-1) or 'synthetic'."""
    try:
        df = load_labels(slug(event))
    except FileNotFoundError:
        return "none"
    non_report = df[df["source"] != "report"]["source"]
    return str(non_report.mode().iloc[0]) if len(non_report) else "unknown"


def load_labels_geojson(event: str, classes: list[str] | None = None,
                        only_flooded: bool = True, precision: int = 5) -> dict:
    """Join labels to segment geometry for the map overlay."""
    from shapely import set_precision

    from aquaroute.db.segments_store import load_segments_gdf

    labels = load_labels(event)
    gdf = load_segments_gdf()
    merged = gdf.merge(labels[["segment_id", "flooded", "depth_proxy", "source"]],
                       on="segment_id", how="inner")
    if only_flooded:
        merged = merged[merged["flooded"]]
    if classes:
        merged = merged[merged["road_class"].astype(str).isin(classes)]

    merged = merged.copy()
    merged["geometry"] = set_precision(merged.geometry.values, grid_size=10 ** (-precision))
    keep = ["segment_id", "road_class", "flooded", "depth_proxy", "source", "geometry"]
    return json.loads(merged[keep].to_json())
