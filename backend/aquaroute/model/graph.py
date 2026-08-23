"""Build the road-*segment* adjacency graph for the GNN (Module 4 / Phase 5).

Nodes are analysis segments (from Phase 2). Two segments are connected when they
are consecutive pieces of the same OSM edge (chain) or when they meet at a shared
OSM node (intersection). This is the graph the GNN propagates flood coupling over
(upstream→downstream). Cached to ``data/segment_graph.npz``.

Public function
---------------
build_segment_edge_index(segments, refresh=False) -> (edge_index[2,E] int64, ids list)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from aquaroute.config import get_config


def _cache_path() -> Path:
    return get_config().cache_dir / "segment_graph.npz"


def build_segment_edge_index(segments, refresh: bool = False):
    """Return an undirected edge_index over segments (in the gdf row order)."""
    ids = segments["segment_id"].astype(str).tolist()
    order = {sid: k for k, sid in enumerate(ids)}
    cache = _cache_path()

    if cache.exists() and not refresh:
        data = np.load(cache, allow_pickle=True)
        if list(data["ids"]) == ids:
            return data["edge_index"], ids

    # Parse "u_v_key_i" → (base="u_v_key", i).
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)  # base -> [(i, idx)]
    node_touch: dict[str, list[int]] = defaultdict(list)          # osm node -> [idx]
    for sid, k in order.items():
        parts = sid.split("_")
        if len(parts) < 4:
            continue
        u, v, _key, i = parts[0], parts[1], parts[2], int(parts[-1])
        base = "_".join(parts[:-1])
        groups[base].append((i, k))

    edges: set[tuple[int, int]] = set()
    for base, items in groups.items():
        items.sort()
        u, v = base.split("_")[0], base.split("_")[1]
        # chain consecutive pieces
        for a in range(len(items) - 1):
            edges.add((items[a][1], items[a + 1][1]))
        # endpoints touch OSM nodes: first piece → u, last piece → v
        first_idx = items[0][1]
        last_idx = items[-1][1]
        node_touch[u].append(first_idx)
        node_touch[v].append(last_idx)

    # Connect segments meeting at a shared OSM node (small cliques; node degree low).
    for _node, seg_list in node_touch.items():
        uniq = list(set(seg_list))
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                edges.add((uniq[a], uniq[b]))

    if edges:
        arr = np.array(sorted(edges), dtype="int64").T  # [2, E]
        # make undirected (both directions)
        edge_index = np.concatenate([arr, arr[::-1]], axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype="int64")

    np.savez(cache, edge_index=edge_index, ids=np.array(ids, dtype=object))
    return edge_index, ids
