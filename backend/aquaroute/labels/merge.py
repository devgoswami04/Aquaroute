"""Merge remote (SAR) labels with crowd (citizen-report) evidence (Module 3).

Citizen reports are higher-trust for the specific segment they name: a 'flooded'
report forces flooded=True, a 'clear' report forces flooded=False, overriding the
SAR label for that segment and recording the source as 'report'.
"""
from __future__ import annotations

import pandas as pd


def merge_report_labels(sar_labels: pd.DataFrame, citizen_reports: pd.DataFrame) -> pd.DataFrame:
    """Return SAR labels with citizen reports applied on top (by segment_id)."""
    out = sar_labels.set_index("segment_id").copy()
    if citizen_reports is None or citizen_reports.empty:
        return out.reset_index()

    for _, r in citizen_reports.iterrows():
        sid = r["segment_id"]
        if sid not in out.index:
            continue
        flooded = str(r["status"]).lower() == "flooded"
        out.loc[sid, "flooded"] = flooded
        out.loc[sid, "source"] = "report"
        if flooded and r.get("depth_est"):
            out.loc[sid, "depth_proxy"] = float(r["depth_est"])
        elif not flooded:
            out.loc[sid, "depth_proxy"] = 0.0
    return out.reset_index()
