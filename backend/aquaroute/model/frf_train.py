"""Train & evaluate the Flood Response Function (Phase 5).

Training regime: the FRF is trained on many **synthetic storms** (varied
onset/duration/intensity) with linear-reservoir depth targets, so the temporal
encoder learns the rainfall-shape → timing mapping. The three real historical
storms (2015/2021/2023) are the held-out validation set — their rainfall never
enters training. Metrics report what the baseline cannot: onset/clearance **timing
error** (hours) and **depth RMSE**, plus an event-level F1 (peak > 0.1 m) against
the Phase-3 SAR/synthetic labels, comparable to Phase 4.

Node features are the static terrain/surface features; rainfall enters only through
the temporal encoder.
"""
from __future__ import annotations

import numpy as np

from aquaroute.model.frf_targets import (
    MAX_DEPTH_M,
    ONSET_THRESHOLD_M,
    derive_events_from_curve,
    get_event_hyetograph,
    random_storm,
    segment_reservoir_params,
    storm_shape,
)

NODE_FEATURES = [
    "length_m", "elevation", "slope", "twi", "depression_depth",
    "upstream_area", "imperviousness", "is_underpass",
]
HORIZON = 24


def prepare_data(propensity_events: list[str] | None = None) -> dict:
    """Assemble node features, graph, per-event targets.

    If real-SAR flood-propensity is available it grounds the target amplitude in
    observed flooding (``propensity_events`` restricts which SAR events train it,
    e.g. to hold one out); otherwise the terrain-susceptibility heuristic is used.
    """
    from aquaroute.config import get_config
    from aquaroute.db.segments_store import load_segments_gdf
    from aquaroute.labels.store import list_labelled_events, load_labels, slug
    from aquaroute.model.graph import build_segment_edge_index

    cfg = get_config()
    seg = load_segments_gdf()
    feats = seg.drop(columns="geometry").copy()
    feats["is_underpass"] = feats["is_underpass"].astype(int)

    edge_index, ids = build_segment_edge_index(seg)
    feats = feats.set_index("segment_id").loc[ids].reset_index()

    raw = feats[NODE_FEATURES].to_numpy(dtype="float64")
    med = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    bad = np.where(~np.isfinite(raw))
    raw[bad] = np.take(med, bad[1])

    params = segment_reservoir_params(feats)   # reservoir dynamics only (shape)

    # Flood extent/magnitude grounded in real SAR (per-segment propensity). This is
    # applied as an amplitude at eval/inference; the FRF itself learns only the
    # normalised temporal shape.
    try:
        from aquaroute.model.propensity import compute_flood_propensity
        p = compute_flood_propensity(propensity_events)
        propensity = feats["segment_id"].map(p).fillna(0.0).to_numpy().astype("float32")
    except Exception:
        propensity = np.full(len(feats), 0.3, dtype="float32")

    ev_meta = {slug(e["name"]): e for e in cfg.events}
    events = {}
    for s in list_labelled_events():
        labels = load_labels(s)
        meta = ev_meta.get(s, {})
        hyeto = get_event_hyetograph(meta.get("start", ""), meta.get("end", ""), HORIZON)
        shape = storm_shape(params, hyeto)                # normalised temporal shape
        flooded = feats["segment_id"].map(
            labels.set_index("segment_id")["flooded"]).fillna(False).to_numpy().astype(int)
        events[s] = {"hyeto": hyeto, "target": shape, "flooded": flooded}

    return {"feats": feats, "ids": ids, "edge_index": edge_index,
            "node_x_raw": raw.astype("float32"), "params": params,
            "propensity": propensity, "events": events}


def _standardize(x: np.ndarray, stats=None):
    if stats is None:
        mean = x.mean(axis=0); std = x.std(axis=0); std[std < 1e-6] = 1.0
        stats = (mean, std)
    mean, std = stats
    return ((x - mean) / std).astype("float32"), stats


def train_frf(data: dict, epochs: int = 40, storms_per_epoch: int = 3,
              hidden: int = 64, gnn: str = "graphsage", temporal: str = "lstm",
              lr: float = 0.01, seed: int = 42, log_every: int = 10, stats=None):
    import torch
    from aquaroute.model.frf import FloodResponseFunction

    x_np, stats = _standardize(data["node_x_raw"], stats)
    x = torch.as_tensor(x_np)
    ei = torch.as_tensor(data["edge_index"], dtype=torch.long)
    params = data["params"]

    model = FloodResponseFunction(len(NODE_FEATURES), HORIZON, hidden, gnn, temporal)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()          # shape targets are dense — plain MSE is fine
    rng = np.random.default_rng(seed)

    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = 0.0
        for _ in range(storms_per_epoch):
            hy = random_storm(rng, HORIZON)
            tgt = torch.as_tensor(storm_shape(params, hy))
            pred = model(x, ei, torch.as_tensor(hy))
            loss = loss + loss_fn(pred, tgt)
        loss.backward()
        opt.step()
        if ep % log_every == 0 or ep == 1:
            print(f"      epoch {ep:>3}/{epochs}  loss={float(loss):.5f}")
    return model, stats


def evaluate_frf(model, data: dict, event: str, stats) -> dict:
    """Depth RMSE + onset/clearance timing MAE (vs physical target) + event F1."""
    from aquaroute.model.evaluate import classification_metrics
    from aquaroute.model.frf import predict_all

    x_np, _ = _standardize(data["node_x_raw"], stats)
    ev = data["events"][event]
    shape_pred = predict_all(model, x_np, data["edge_index"], ev["hyeto"])  # [N,T] shape
    shape_true = ev["target"]

    # depth = MAX_DEPTH · propensity(real SAR) · shape(FRF)
    amp = (MAX_DEPTH_M * data["propensity"])[:, None]
    pred = amp * shape_pred
    target = amp * shape_true

    depth_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))

    # Timing on segments that actually flood (amplitude clears the onset threshold).
    tgt_peak = target.max(axis=1)
    idx = np.where(tgt_peak > ONSET_THRESHOLD_M)[0]
    if idx.size > 4000:
        idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
    onset_err = clear_err = 0.0
    n = 0
    for i in idx:
        pe = derive_events_from_curve(pred[i])
        te = derive_events_from_curve(target[i])
        if pe["onset"] is not None and te["onset"] is not None:
            onset_err += abs(pe["onset"] - te["onset"])
            clear_err += abs((pe["clearance"] or 0) - (te["clearance"] or 0))
            n += 1

    peak_pred = pred.max(axis=1)
    pred_flood = (peak_pred > ONSET_THRESHOLD_M).astype(int)
    clsm = classification_metrics(ev["flooded"], pred_flood, peak_pred)

    return {
        "event": event,
        "depth_rmse_m": round(depth_rmse, 4),
        "onset_mae_h": round(onset_err / n, 3) if n else None,
        "clearance_mae_h": round(clear_err / n, 3) if n else None,
        "event_f1": clsm["f1"],
        "event_roc_auc": clsm.get("roc_auc"),
    }
