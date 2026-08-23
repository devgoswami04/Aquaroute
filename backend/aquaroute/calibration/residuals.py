"""Per-segment residuals: predicted minus observed depth (Module 5)."""
from __future__ import annotations

import pandas as pd


def compute_residuals(predictions, observations: pd.DataFrame) -> pd.DataFrame:
    """residual = predicted − observed depth, per segment.

    ``predictions`` may be a dict {segment_id: depth} or a DataFrame with
    columns [segment_id, pred]. ``observations`` has [segment_id, depth_obs].
    Only segments present in *both* are returned (we can only calibrate what we
    observe).
    """
    if isinstance(predictions, dict):
        preds = pd.DataFrame({"segment_id": list(predictions.keys()),
                              "pred": list(predictions.values())})
    else:
        preds = predictions.rename(columns={"peak_depth": "pred"})[["segment_id", "pred"]]

    obs = observations[["segment_id", "depth_obs"]].rename(columns={"depth_obs": "obs"})
    merged = preds.merge(obs, on="segment_id", how="inner")
    merged["residual"] = merged["pred"] - merged["obs"]
    return merged
