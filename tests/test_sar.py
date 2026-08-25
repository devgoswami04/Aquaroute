"""Tests for real Sentinel-1 SAR flood detection.

The change-detection core runs on synthetic dB arrays (no network). A network-
guarded test hits Planetary Computer for a real scene when available.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

network = pytest.mark.skipif(
    os.environ.get("AQUAROUTE_SKIP_NETWORK") == "1",
    reason="network disabled via AQUAROUTE_SKIP_NETWORK=1",
)


def test_detect_water_change_and_absolute():
    from aquaroute.ingestion.sar import detect_water

    rng = np.random.default_rng(0)
    ref = 20 + rng.normal(0, 0.5, (60, 60))    # dry: ~20 dB everywhere
    dur = ref.copy()
    dur[10:25, 10:25] -= 6.0                    # a block floods: -6 dB drop
    water = detect_water(dur, ref, change_db=3.0)
    # the flooded block is detected...
    assert water[12:23, 12:23].mean() > 0.9
    # ...and stable dry ground is (mostly) not water
    assert water[40:55, 40:55].mean() < 0.15


def test_detect_water_handles_nan():
    from aquaroute.ingestion.sar import detect_water

    ref = np.full((20, 20), 20.0); dur = np.full((20, 20), 20.0)
    ref[0, 0] = np.nan; dur[0, 0] = np.nan
    w = detect_water(dur, ref)
    assert w.shape == (20, 20) and w[0, 0] == 0     # nan pixel not water


@network
def test_real_sar_mask_from_planetary_computer(tmp_path):
    from aquaroute.ingestion.sar import SarUnavailable, fetch_sar_flood_mask

    try:
        p = fetch_sar_flood_mask("2021-11-06", "2021-11-12", out_tif=tmp_path / "sar.tif")
    except SarUnavailable:
        pytest.skip("no SAR scene / PC unreachable")
    import json
    import rasterio

    meta = json.loads((p.with_suffix(".json")).read_text())
    assert meta["source"] == "planetary-computer"
    with rasterio.open(p) as s:
        a = s.read(1)
    frac = float((a > 0).mean())
    assert 0.0 < frac < 0.5      # a plausible flood-water fraction, not all/nothing
