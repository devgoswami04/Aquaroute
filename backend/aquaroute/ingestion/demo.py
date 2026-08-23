"""Phase 1 verification demo.

Run with ``python -m aquaroute.ingestion.demo`` (or ``make ingest``). Prints a
24-hour precipitation series for the bbox centre and the edge/node counts of the
downloaded OSM road graph. DEM download is attempted only if an OpenTopography
key is configured, so the demo runs end-to-end even without one.
"""
from __future__ import annotations

import sys

from aquaroute.config import get_config
from aquaroute.ingestion import (
    fetch_dem,
    fetch_rainfall_forecast,
    fetch_road_graph,
)


def main() -> int:
    cfg = get_config()
    print(f"AquaRoute Phase 1 ingestion demo — bbox {cfg.bbox}\n")

    # --- Rainfall (Open-Meteo, keyless) ---
    print("[1/3] Fetching 24 h rainfall forecast (Open-Meteo)...")
    rain = fetch_rainfall_forecast(hours=24)
    hourly = rain[rain["resolution"] == "hourly"]
    centre_id = hourly["point_id"].iloc[0] if not hourly.empty else None
    series = hourly[hourly["point_id"] == centre_id].sort_values("time")
    print(f"      grid points: {rain['point_id'].nunique()}  "
          f"rows: {len(rain)} (hourly + 15-min)")
    print(f"      24 h hourly precipitation at {centre_id} (mm):")
    for _, r in series.iterrows():
        bar = "#" * int(round((r['precip_mm'] or 0) * 4))
        print(f"        {r['time']:%Y-%m-%d %H:%M}  {r['precip_mm']:5.2f}  {bar}")
    print(f"      forecast total: {series['precip_mm'].sum():.2f} mm\n")

    # --- Road graph (OSMnx, keyless) ---
    print("[2/3] Downloading drivable road graph (OSMnx)... (first run is slow)")
    graph = fetch_road_graph()
    print(f"      nodes: {graph.number_of_nodes()}  edges (segments): "
          f"{graph.number_of_edges()}\n")

    # --- DEM (OpenTopography, free key) ---
    print("[3/3] Fetching DEM (OpenTopography)...")
    try:
        dem_path = fetch_dem()
        print(f"      DEM saved: {dem_path}")
    except RuntimeError as e:
        print(f"      SKIPPED — {e}")

    print("\nPhase 1 OK. Next: Phase 2 (pysheds DEM conditioning + road "
          "segmentation into PostGIS).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
