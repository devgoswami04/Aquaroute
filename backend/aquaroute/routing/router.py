"""Vehicle-aware, time-expanded safe routing (Module 6).

Edges carry a 24 h predicted depth curve (from the FRF, after self-calibration).
An edge is *impassable* to a vehicle when the water at the vehicle's arrival hour
exceeds its depth threshold — so passability varies over the forecast horizon and
by vehicle class. ``safe_route`` runs a time-dependent Dijkstra (a materialised
time-expanded search) minimising **routing regret** = travel time + flood-exposure
penalty, not raw distance. We reuse NetworkX graph structures rather than
hand-rolling the search badly (brief §2.1, §6).
"""
from __future__ import annotations

import functools
import heapq
import math

import numpy as np

from aquaroute.routing.vehicle import get_threshold

# Class → assumed speed (km/h) for travel-time estimates.
_SPEED = {"motorway": 60, "trunk": 50, "primary": 40, "secondary": 35,
          "tertiary": 30, "residential": 22, "living_street": 15, "service": 18,
          "unclassified": 25}
RISK_WEIGHT = 4.0          # how strongly wet-but-passable edges are penalised
WET = 0.05                 # depth (m) above which an edge counts as "wet"
HORIZON = 24


def _first(x):
    return x[0] if isinstance(x, list) else x


def _speed(highway) -> float:
    return _SPEED.get(str(_first(highway)), 25)


class RouterContext:
    """Routing graph for one forecast scenario, with per-edge depth curves."""

    def __init__(self, scenario: str = "live"):
        import networkx as nx

        from aquaroute.calibration.engine import get_engine
        from aquaroute.ingestion.roads import fetch_road_graph
        from aquaroute.model.predictor import get_predictor

        self.scenario = scenario
        pred = get_predictor()
        entry = pred.predict_current(scenario=scenario)
        alpha = get_engine().alpha_vector(pred.ids)[:, None]
        depth = entry["depth"] * alpha                      # calibrated [N,24]

        # Max depth curve per OSM edge (u,v,key) across its analysis segments.
        edge_depth: dict[tuple, np.ndarray] = {}
        for i, sid in enumerate(pred.ids):
            p = sid.split("_")
            key = (int(p[0]), int(p[1]), int(p[2]))
            cur = edge_depth.get(key)
            edge_depth[key] = depth[i] if cur is None else np.maximum(cur, depth[i])

        g0 = fetch_road_graph()
        G = nx.DiGraph()
        self.node_xy = {n: (d["x"], d["y"]) for n, d in g0.nodes(data=True)}
        for u, v, k, data in g0.edges(keys=True, data=True):
            length = float(data.get("length", 0.0) or 0.0)
            if length <= 0:
                continue
            dc = edge_depth.get((u, v, k))
            if dc is None:
                dc = np.zeros(HORIZON, dtype="float32")
            base_time = (length / 1000.0) / _speed(data.get("highway")) * 60.0  # minutes
            if G.has_edge(u, v) and G[u][v]["length"] <= length:
                continue  # keep the shorter of parallel edges
            G.add_edge(u, v, length=length, base_time=base_time, depth=dc)
        self.G = G
        self._nodes = np.array(list(self.node_xy.keys()))
        self._coords = np.array([self.node_xy[n] for n in self._nodes])  # (lon,lat)

    # --- geocoding-lite: snap a lon/lat to the nearest graph node ---
    def nearest_node(self, lon: float, lat: float):
        d = (self._coords[:, 0] - lon) ** 2 + (self._coords[:, 1] - lat) ** 2
        return self._nodes[int(np.argmin(d))]

    # --- searches ---
    def _time_dependent_dijkstra(self, src, dst, depart_hour, threshold,
                                 risk_weight, avoid_impassable):
        G = self.G
        best = {src: 0.0}
        arr = {src: 0.0}
        prev = {}
        pq = [(0.0, 0.0, src)]
        while pq:
            cost, t, u = heapq.heappop(pq)
            if u == dst:
                break
            if cost > best.get(u, math.inf):
                continue
            for v in G.successors(u):
                e = G[u][v]
                h = int(min(HORIZON - 1, max(0, depart_hour + t / 60.0)))
                depth = float(e["depth"][h])
                if avoid_impassable and depth >= threshold:
                    continue
                risk = min(depth / threshold, 1.0) if threshold > 0 else 0.0
                over = max(0.0, depth - threshold)
                step = e["base_time"] * (1 + risk_weight * risk) + 1000.0 * over
                nt = t + e["base_time"]
                nc = cost + step
                if nc < best.get(v, math.inf):
                    best[v] = nc
                    arr[v] = nt
                    prev[v] = u
                    heapq.heappush(pq, (nc, nt, v))
        if dst not in prev and dst != src:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _shortest_by_length(self, src, dst):
        import networkx as nx
        try:
            return nx.shortest_path(self.G, src, dst, weight="length")
        except nx.NetworkXNoPath:
            return None

    # --- route evaluation ---
    def _evaluate(self, path, depart_hour, threshold):
        if not path or len(path) < 2:
            return None
        length = time = exposure = blocked = 0.0
        max_depth = 0.0
        coords = [list(self.node_xy[path[0]])]
        blocked_hours = []
        for u, v in zip(path, path[1:]):
            e = self.G[u][v]
            h = int(min(HORIZON - 1, max(0, depart_hour + time / 60.0)))
            depth = float(e["depth"][h])
            length += e["length"]
            time += e["base_time"]
            max_depth = max(max_depth, depth)
            if depth > WET:
                exposure += e["length"]
            if depth >= threshold:
                blocked += e["length"]
                blocked_hours.append(h)
            coords.append(list(self.node_xy[v]))
        return {
            "nodes": len(path),
            "distance_m": round(length, 1),
            "time_min": round(time, 1),
            "max_depth_m": round(max_depth, 3),
            "exposure_m": round(exposure, 1),
            "blocked_m": round(blocked, 1),
            "blocked_hours": sorted(set(blocked_hours)),
            "geometry": {"type": "LineString", "coordinates": coords},
        }

    def route(self, origin, dest, depart_hour: int, vehicle: str) -> dict:
        """Return safe vs shortest routes + a plain-language advisory."""
        threshold = get_threshold(vehicle)
        o = self.nearest_node(*origin)
        d = self.nearest_node(*dest)

        safe_path = self._time_dependent_dijkstra(
            o, d, depart_hour, threshold, RISK_WEIGHT, avoid_impassable=True)
        note = None
        if safe_path is None:  # no fully-passable route → allow, but flag
            safe_path = self._time_dependent_dijkstra(
                o, d, depart_hour, threshold, RISK_WEIGHT, avoid_impassable=False)
            note = "No fully passable route found; safest available route shown."

        shortest_path = self._shortest_by_length(o, d)
        safe = self._evaluate(safe_path, depart_hour, threshold)
        shortest = self._evaluate(shortest_path, depart_hour, threshold)
        return {
            "vehicle": vehicle,
            "threshold_m": threshold,
            "depart_hour": depart_hour,
            "scenario": self.scenario,
            "origin_node": int(o), "dest_node": int(d),
            "safe_route": safe,
            "shortest_route": shortest,
            "advisory": _advisory(vehicle, threshold, safe, shortest, note),
        }


def _fmt_window(hours, depart_hour):
    if not hours:
        return ""
    lo, hi = min(hours), max(hours)
    return f"{lo:02d}:00-{hi + 1:02d}:00"


def _advisory(vehicle, threshold, safe, shortest, note) -> str:
    v = vehicle.replace("_", "-")
    if shortest and shortest["blocked_m"] > 0:
        win = _fmt_window(shortest["blocked_hours"], 0)
        extra = (safe["distance_m"] - shortest["distance_m"]) if safe and shortest else 0
        msg = (f"Shortest route crosses water impassable to a {v} "
               f"(>={threshold:.2f} m{', ' + win if win else ''}). "
               f"Safe route detours +{max(0, extra):.0f} m to avoid it.")
    elif safe and safe["max_depth_m"] > WET:
        msg = (f"Route passable for a {v}; some standing water up to "
               f"{safe['max_depth_m']:.2f} m — drive with caution.")
    else:
        msg = f"Route clear for a {v}."
    return (note + " " + msg) if note else msg


@functools.lru_cache(maxsize=4)
def get_router(scenario: str = "live") -> RouterContext:
    return RouterContext(scenario)


# --- brief §6 signatures (thin wrappers over the cached RouterContext) ---

def build_time_expanded_graph(scenario: str = "live") -> RouterContext:
    """Build (or fetch cached) the time-expanded routing graph for a scenario."""
    return get_router(scenario)


def safe_route(origin, dest, depart_time: int, vehicle: str,
               scenario: str = "live") -> dict:
    """Risk-aware route for a vehicle class departing at ``depart_time`` (hour)."""
    return get_router(scenario).route(origin, dest, depart_time, vehicle)
