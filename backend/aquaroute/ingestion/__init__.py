"""Module 1 — data ingestion.

Phase 1 implements the three keyless/free-key backbone feeds:
rainfall (Open-Meteo), road graph (OSMnx), DEM (OpenTopography). Land-use, SAR
labels and the sensor/traffic/works feed adapters land in later phases (brief §6).
"""
from aquaroute.ingestion.dem import fetch_dem
from aquaroute.ingestion.rainfall import fetch_rainfall_forecast, fetch_rainfall_history
from aquaroute.ingestion.roads import fetch_road_graph

__all__ = [
    "fetch_rainfall_forecast",
    "fetch_rainfall_history",
    "fetch_road_graph",
    "fetch_dem",
]
