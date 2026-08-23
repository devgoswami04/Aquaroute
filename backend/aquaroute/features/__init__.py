"""Module 2 — preprocessing & feature engineering.

DEM conditioning (pysheds), road segmentation, and per-segment feature vectors.
"""
from aquaroute.features.hydrology import HydroLayers, condition_dem, sample_layers_at
from aquaroute.features.segment_features import (
    build_rainfall_descriptors,
    build_segment_features,
)
from aquaroute.features.segmentation import segment_road_graph

__all__ = [
    "condition_dem",
    "sample_layers_at",
    "HydroLayers",
    "segment_road_graph",
    "build_segment_features",
    "build_rainfall_descriptors",
]
