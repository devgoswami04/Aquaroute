"""OpenRouteService avoid-polygons fallback / baseline (Module 6).

A simple alternative to the custom time-expanded search: ask ORS to route while
avoiding polygons around currently-flooded areas. Needs a free ORS key
(``ORS_API_KEY``); raises a clear error if absent so the primary router is used.
"""
from __future__ import annotations

from aquaroute.config import get_config


def ors_avoid_fallback(origin, dest, flooded_polys, profile: str = "driving-car") -> dict:
    """Route origin→dest avoiding ``flooded_polys`` (list of GeoJSON Polygons).

    ``origin``/``dest`` are (lon, lat). Returns the ORS GeoJSON route.
    """
    key = get_config().settings.ors_api_key
    if not key:
        raise RuntimeError(
            "ORS_API_KEY is not set. Add a free key from openrouteservice.org to .env "
            "to use the avoid-polygons fallback, or use the built-in safe_route."
        )
    import openrouteservice as ors

    client = ors.Client(key=key)
    params = {
        "coordinates": [list(origin), list(dest)],
        "profile": profile,
        "format": "geojson",
    }
    if flooded_polys:
        params["options"] = {"avoid_polygons": {
            "type": "MultiPolygon",
            "coordinates": [p["coordinates"] for p in flooded_polys],
        }}
    return client.directions(**params)
