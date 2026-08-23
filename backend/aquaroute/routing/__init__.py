"""Module 6 — vehicle-class-aware safe routing over a time-expanded graph."""
from aquaroute.routing.ors import ors_avoid_fallback
from aquaroute.routing.router import (
    RouterContext,
    build_time_expanded_graph,
    get_router,
    safe_route,
)
from aquaroute.routing.vehicle import get_threshold, vehicle_thresholds

__all__ = [
    "vehicle_thresholds",
    "get_threshold",
    "build_time_expanded_graph",
    "safe_route",
    "get_router",
    "RouterContext",
    "ors_avoid_fallback",
]
