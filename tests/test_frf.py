"""Phase 5 tests — Flood Response Function.

The target-synthesis and event-derivation tests run without torch. The model
tests skip cleanly if torch / torch_geometric aren't installed, and a further
test checks the saved FRF checkpoint if the pipeline has been run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FRF_CKPT = REPO / "data" / "models" / "frf.pt"


def test_derive_events_shapes():
    from aquaroute.model.frf_targets import derive_events_from_curve

    # rise then fall, peak at index 3
    curve = np.array([0.0, 0.05, 0.2, 0.5, 0.3, 0.08, 0.0])
    ev = derive_events_from_curve(curve, thresh=0.1)
    assert ev["peak"] == 3 and abs(ev["peak_depth"] - 0.5) < 1e-6
    assert ev["onset"] == 2                # first > 0.1
    assert ev["clearance"] == 5            # first <= 0.1 after peak
    dry = derive_events_from_curve(np.zeros(7))
    assert dry["onset"] is None and dry["clearance"] is None


def test_depth_scales_with_susceptibility_and_storm():
    from aquaroute.model.frf_targets import synthesize_depth_curves

    # Two segments differing only in susceptibility; same storm.
    feats = pd.DataFrame({
        "segment_id": ["hi", "lo"],
        "imperviousness": [0.9, 0.9], "upstream_area": [100.0, 100.0],
        "depression_depth": [5.0, 5.0], "slope": [0.01, 0.01],
        "twi": [18.0, 18.0], "susceptibility": [0.95, 0.1],
    })
    storm = np.array([0, 2, 5, 8, 4, 1, 0] + [0] * 17, dtype=float)
    curves = synthesize_depth_curves(feats, storm)
    assert curves.shape == (2, 24)
    assert curves[0].max() > curves[1].max()          # higher susceptibility → deeper

    big = synthesize_depth_curves(feats, storm * 4)
    assert big[0].max() >= curves[0].max()            # bigger storm → deeper (or saturated)


def _have_torch():
    try:
        import torch  # noqa
        import torch_geometric  # noqa
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_torch(), reason="torch/PyG not installed")
def test_frf_forward_shapes():
    import torch
    from aquaroute.model.frf import FloodResponseFunction

    N, F, T = 20, 8, 24
    x = torch.randn(N, F)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    hyeto = torch.rand(T)
    model = FloodResponseFunction(F, T, hidden=16)
    out = model(x, ei, hyeto)
    assert out.shape == (N, T)
    assert (out >= 0).all()  # softplus


@pytest.mark.skipif(not FRF_CKPT.exists(), reason="FRF not trained")
def test_frf_checkpoint_loads_and_predicts():
    import torch
    from aquaroute.model.frf import FloodResponseFunction, predict_all

    ck = torch.load(FRF_CKPT, weights_only=False)
    model = FloodResponseFunction(len(ck["node_features"]), ck["horizon"],
                                  ck["hidden"], ck["gnn"], ck["temporal"])
    model.load_state_dict(ck["state_dict"])
    x = np.random.randn(30, len(ck["node_features"])).astype("float32")
    ei = np.array([[0, 1, 2], [1, 2, 0]])
    out = predict_all(model, x, ei, np.random.rand(ck["horizon"]).astype("float32"))
    assert out.shape == (30, ck["horizon"])
