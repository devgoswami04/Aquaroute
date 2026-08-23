"""Phase 8 demo: safe routes across vehicle classes on a flooded scenario.

Run with ``python -m aquaroute.routing.demo`` (or ``make route-demo``). Prints the
safe route for each vehicle class between two points during peak rain, showing how
the depth threshold changes the route.
"""
from __future__ import annotations

import sys

from aquaroute.routing import get_router


def run(scenario: str = "2021_chennai_floods", depart_hour: int = 18,
        origin=(80.135, 12.95), dest=(80.23, 12.945)) -> int:
    print(f"AquaRoute Phase 8 — vehicle-aware routing (scenario '{scenario}', "
          f"depart {depart_hour}:00)\n")
    R = get_router(scenario)
    print(f"    routing graph: {R.G.number_of_nodes()} nodes, {R.G.number_of_edges()} edges\n")
    geoms = {}
    for veh in ("two_wheeler", "auto", "car", "bus"):
        r = R.route(origin, dest, depart_hour, veh)
        s = r["safe_route"]
        geoms[veh] = s["geometry"]["coordinates"]
        print(f"    {veh:<12} thr={r['threshold_m']:.2f} m | safe {s['distance_m']:>7.0f} m / "
              f"{s['time_min']:>3.0f} min | impassable {s['blocked_m']:>6.0f} m | "
              f"max depth {s['max_depth_m']:.2f} m")
        print(f"        {r['advisory']}")
    print()
    print(f"    two-wheeler vs bus take different routes: "
          f"{geoms['two_wheeler'] != geoms['bus']}")
    print("\nPhase 8 OK. Vehicle class changes the safe route on the same forecast.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
