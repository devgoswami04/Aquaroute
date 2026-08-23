"""Flood Response Function — the novel model (Module 4, Phase 5).

Architecture:
  * **Temporal encoder** (LSTM or TCN) over the rainfall hyetograph → an event
    temporal embedding.
  * **GNN** (PyG GraphSAGE or GAT) over the road-segment graph → per-segment
    spatial embedding that propagates upstream→downstream coupling.
  * **Decoder** (MLP) maps [spatial ⊕ temporal] → a depth-vs-time curve per segment.

We reuse PyG's graph layers and torch's recurrent/conv ops; only the composition
and the flood-specific decoding are ours (brief §2.1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

from aquaroute.model.frf_targets import derive_events_from_curve


class TemporalEncoder(nn.Module):
    """Encode a length-T hyetograph into a fixed embedding."""

    def __init__(self, hidden: int, kind: str = "lstm"):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        elif kind == "tcn":
            self.tcn = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4), nn.ReLU(),
            )
        else:
            raise ValueError(f"unknown temporal encoder: {kind!r}")

    def forward(self, hyeto: torch.Tensor) -> torch.Tensor:
        # hyeto: [T] -> embedding [hidden]
        x = hyeto.view(1, -1, 1)  # [B=1, T, 1]
        if self.kind == "lstm":
            _, (h, _) = self.lstm(x)
            return h[-1].squeeze(0)  # [hidden]
        y = self.tcn(x.transpose(1, 2))  # [1, hidden, T]
        return y.mean(dim=2).squeeze(0)  # global average pool -> [hidden]


class FloodResponseFunction(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 64,
                 gnn: str = "graphsage", temporal: str = "lstm"):
        super().__init__()
        self.horizon = horizon
        self.temporal = TemporalEncoder(hidden, temporal)
        Conv = GATConv if gnn in ("gat",) else SAGEConv
        self.gnn1 = Conv(n_features, hidden)
        self.gnn2 = Conv(hidden, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                hyeto: torch.Tensor) -> torch.Tensor:
        e = self.temporal(hyeto)                 # [hidden]
        z = F.relu(self.gnn1(x, edge_index))
        z = F.relu(self.gnn2(z, edge_index))     # [N, hidden]
        e_b = e.unsqueeze(0).expand(z.size(0), -1)
        out = self.decoder(torch.cat([z, e_b], dim=1))  # [N, T]
        return F.softplus(out)                   # depths >= 0


@torch.no_grad()
def predict_all(model: FloodResponseFunction, x, edge_index, hyeto) -> np.ndarray:
    """Depth-vs-time [N, T] for every segment (numpy)."""
    model.eval()
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    ei = torch.as_tensor(np.asarray(edge_index), dtype=torch.long)
    ht = torch.as_tensor(np.asarray(hyeto), dtype=torch.float32)
    return model(xt, ei, ht).cpu().numpy()


def predict_curve(depth_row: np.ndarray) -> dict:
    """Format a single segment's curve as {t:[...], depth:[...]} for the chart/API."""
    depth_row = np.asarray(depth_row)
    return {"t": list(range(len(depth_row))), "depth": [round(float(d), 3) for d in depth_row]}


def derive_events(depth_row: np.ndarray) -> dict:
    """Onset / peak / clearance (hours) + peak depth from a predicted curve."""
    return derive_events_from_curve(depth_row)
