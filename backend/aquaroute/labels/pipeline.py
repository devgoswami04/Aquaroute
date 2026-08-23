"""Phase 3 pipeline: build per-event flood labels for every configured event.

Uses GEE Sentinel-1 SAR if EE_PROJECT is set, else the synthetic terrain+rainfall
mask. Applies synthetic citizen reports (to demo the merge), and saves labels.
Run with ``python -m aquaroute.labels.pipeline`` (or ``make labels``).
"""
from __future__ import annotations

import sys
from datetime import date

from aquaroute.config import get_config
from aquaroute.db.segments_store import load_segments_gdf
from aquaroute.features.hydrology import condition_dem
from aquaroute.ingestion import fetch_dem
from aquaroute.labels.merge import merge_report_labels
from aquaroute.labels.sar_labels import label_event_from_sar
from aquaroute.labels.store import save_labels, slug
from aquaroute.synthetic.flood_labels import (
    severity_from_rainfall,
    synthesize_sar_mask,
    synthetic_citizen_reports,
)


def _ee_available() -> bool:
    if not get_config().settings.ee_project:
        return False
    try:
        import ee  # noqa: F401
        return True
    except Exception:
        return False


def _event_total_mm(start: str, end: str) -> float:
    try:
        from aquaroute.ingestion.rainfall import fetch_rainfall_history
        rain = fetch_rainfall_history(start=start, end=end)
        hourly = rain[rain["resolution"] == "hourly"]
        per_point = hourly.groupby("point_id")["precip_mm"].sum()
        return float(per_point.mean())
    except Exception:
        return 150.0  # fallback severity driver


def run() -> list:
    cfg = get_config()
    print("AquaRoute Phase 3 — flood labelling\n")

    dem = fetch_dem()
    print("[*] Conditioning DEM for depth proxy / synthetic masks...")
    layers = condition_dem(dem)
    segments = load_segments_gdf()
    use_ee = _ee_available()
    print(f"[*] SAR source: {'Google Earth Engine (Sentinel-1)' if use_ee else 'SYNTHETIC fallback'}")

    summaries = []
    for ev in cfg.events:
        name = ev["name"]
        mid = _mid_date(ev["start"], ev["end"])
        total_mm = _event_total_mm(ev["start"], ev["end"])
        sev = severity_from_rainfall(total_mm)
        print(f"\n== {name}  (rain~{total_mm:.0f}mm, flood-fraction~{sev:.2f}) ==")

        if use_ee:
            from aquaroute.ingestion.sar import fetch_sar_flood_mask
            mask = fetch_sar_flood_mask(mid)
            source = "sar"
        else:
            mask = cfg.cache_dir / f"synmask_{slug(name)}.tif"
            synthesize_sar_mask(name, layers, mask, sev)
            source = "synthetic"

        labels = label_event_from_sar(mask, segments, layers, source=source)
        reports = synthetic_citizen_reports(segments, name)
        labels = merge_report_labels(labels, reports)

        summary = save_labels(name, labels)
        summaries.append(summary)
        print(f"   flooded segments: {summary['flooded']} / {summary['count']}  "
              f"(source={source}, +{len(reports)} citizen reports)  -> {summary['file']}")

    print("\nPhase 3 OK. Overlay on the map: GET /labels?event=<name> ; "
          "pick an event in the UI dropdown.")
    return summaries


def _mid_date(start: str, end: str) -> str:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    return date.fromordinal((s.toordinal() + e.toordinal()) // 2).isoformat()


if __name__ == "__main__":
    run()
    sys.exit(0)
