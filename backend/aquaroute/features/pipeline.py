"""Phase 2 pipeline: DEM -> hydrology -> segments -> features -> store.

Run with ``python -m aquaroute.features.pipeline`` (or ``make features``). Produces
the per-segment feature table, writes it to PostGIS (if running) and always to
``data/segments.geojson`` for the map. Verification step for brief Phase 2.
"""
from __future__ import annotations

import sys

from aquaroute.config import get_config
from aquaroute.db.segments_store import save_segments
from aquaroute.features import (
    build_segment_features,
    condition_dem,
    segment_road_graph,
)
from aquaroute.ingestion import fetch_dem, fetch_road_graph


def run() -> dict:
    cfg = get_config()
    print(f"AquaRoute Phase 2 — feature pipeline over {cfg.bbox}\n")

    print("[1/5] DEM (OpenTopography key if set, else keyless AWS Terrain Tiles)...")
    dem_path = fetch_dem()
    print(f"      DEM: {dem_path}")

    print("[2/5] Conditioning DEM (pysheds: fill -> flow dir -> accumulation, TWI)...")
    layers = condition_dem(dem_path)
    print(f"      grid {layers.nrows}x{layers.ncols}, cell ~{layers.cell_width_m:.0f} m; "
          f"TWI range [{layers.twi.min():.2f}, {layers.twi.max():.2f}]")

    print("[3/5] Road graph (OSMnx, cached)...")
    graph = fetch_road_graph()
    print(f"      edges: {graph.number_of_edges()}")

    print("[4/5] Segmenting road graph (~100 m) + building per-segment features...")
    segments = segment_road_graph(graph)
    segments = build_segment_features(segments, layers)
    print(f"      segments: {len(segments)}")
    print(f"      mean length: {segments['length_m'].mean():.1f} m; "
          f"underpasses: {int(segments['is_underpass'].sum())}")
    top = segments.nlargest(5, "susceptibility")[["segment_id", "road_class",
                                                  "twi", "depression_depth",
                                                  "susceptibility"]]
    print("      top-5 susceptible segments (heuristic, pre-model):")
    for _, r in top.iterrows():
        print(f"        {r['segment_id']:>22}  {str(r['road_class'])[:12]:12}  "
              f"TWI={r['twi']:.2f}  depr={r['depression_depth']:.2f}  "
              f"score={r['susceptibility']:.2f}")

    print("[5/5] Saving segments (PostGIS if up, always GeoJSON cache)...")
    summary = save_segments(segments)
    print(f"      {summary}")

    print("\nPhase 2 OK. Start the API and open the map:")
    print("  uvicorn aquaroute.api.main:app --app-dir backend --port 8000")
    print("  cd frontend && npm run dev   # /segments renders on Leaflet")
    return summary


if __name__ == "__main__":
    run()
    sys.exit(0)
