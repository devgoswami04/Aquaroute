"""Phase 10 tests — evaluation report helpers & artifacts."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_md_table():
    from aquaroute.eval.report import _md_table

    md = _md_table(["A", "B"], [[1, 2], [3, 4]])
    lines = md.strip().splitlines()
    assert lines[0] == "| A | B |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | 2 |"


@pytest.mark.skipif(not (REPO / "docs" / "EVALUATION.md").exists(),
                    reason="eval report not generated (run `make eval`)")
def test_eval_artifacts_present():
    figs = REPO / "docs" / "figures"
    for f in ["01_baseline_vs_frf.png", "02_frf_per_event.png",
              "03_self_calibration.png", "04_routing_regret.png"]:
        assert (figs / f).exists() and (figs / f).stat().st_size > 1000
    assert "Evaluation" in (REPO / "docs" / "EVALUATION.md").read_text(encoding="utf-8")
