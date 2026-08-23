"""The 'road-fixed' test (brief §8, Phase 7 verification).

Take chronically-flooded segments, then partway through a sequence of events
'repair' two of them — one with a public-works event (explicit reset), one
silently (only the sensor/traffic observations change, so the change-point
detector must catch it). Show that each repaired segment's flood prediction
**retires within a few events**, while a control segment stays flooded.

Run: ``python -m aquaroute.calibration.roadfixed`` (or ``make calibrate``).
"""
from __future__ import annotations

import sys

import numpy as np

from aquaroute.calibration.engine import RETIRE_THRESHOLD, CalibrationEngine
from aquaroute.calibration.observations import normalize_observations
from aquaroute.synthetic.feeds import (
    poll_public_works,
    stream_sensor_readings,
    stream_traffic_flow,
)


def run_road_fixed_test(n_events: int = 8, repair_at: int = 3, scenario: str = "2021_chennai_floods"):
    from aquaroute.config import get_config
    from aquaroute.model.predictor import get_predictor

    print("AquaRoute Phase 7 — self-calibration 'road-fixed' test\n")
    print(f"[*] Loading FRF predictions for scenario '{scenario}'...")
    pred = get_predictor()
    entry = pred.predict_current(scenario=scenario)
    ids = pred.ids
    peaks = entry["events"]["peak_depth"].to_numpy()

    # Chronically-flooded candidates: the deepest-predicted segments.
    chronic = [ids[i] for i in np.argsort(-peaks)[:40]]
    peak_of = {sid: float(peaks[pred.id_to_idx[sid]]) for sid in chronic}
    repaired_works = chronic[0]     # fixed WITH a public-works event
    repaired_silent = chronic[1]    # fixed silently (change-point must catch it)
    control = chronic[2]            # stays flooded throughout
    tracked = chronic               # observe all of them each event

    print(f"    tracking {len(tracked)} chronically-flooded segments")
    print(f"    repaired (works event): {repaired_works}  (FRF peak {peak_of[repaired_works]:.2f} m)")
    print(f"    repaired (silent):      {repaired_silent}  (FRF peak {peak_of[repaired_silent]:.2f} m)")
    print(f"    control (stays wet):    {control}  (FRF peak {peak_of[control]:.2f} m)\n")

    # Isolated engine so the demo doesn't touch the live API state.
    eng = CalibrationEngine(path=get_config().cache_dir / "_roadfixed_state.json")
    eng.reset_all()
    pred_dict = {sid: peak_of[sid] for sid in tracked}

    def corrected(sid):
        return eng.alpha(sid) * peak_of[sid]

    print(f"    {'event':>5} | {'works-fix depth':>15} | {'silent-fix depth':>16} | {'control depth':>13}")
    print("    " + "-" * 60)
    retired = {"works": None, "silent": None}
    traj = {"works": [], "silent": [], "control": []}
    for e in range(n_events):
        fixed = e >= repair_at
        true_depth = []
        for sid in tracked:
            d = peak_of[sid]
            if fixed and sid in (repaired_works, repaired_silent):
                d = 0.0            # the road no longer floods
            true_depth.append(d)

        # Feeds (identical contract to a real broker).
        sensors = stream_sensor_readings(tracked, true_depth, ts=f"evt{e}", seed=e)
        traffic = stream_traffic_flow(tracked, true_depth, ts=f"evt{e}", seed=e)
        obs = normalize_observations(sensors + traffic, rain_intensity=20.0)

        works_ids = []
        if e == repair_at:
            works = poll_public_works([repaired_works], ts=f"evt{e}")
            works_ids = [w["segment_id"] for w in works]

        eng.run_cycle(pred_dict, observations=obs, works_ids=works_ids, rain_intensity=20.0)

        cw, cs, cc = corrected(repaired_works), corrected(repaired_silent), corrected(control)
        traj["works"].append(cw); traj["silent"].append(cs); traj["control"].append(cc)
        if retired["works"] is None and fixed and cw < RETIRE_THRESHOLD:
            retired["works"] = e - repair_at
        if retired["silent"] is None and fixed and cs < RETIRE_THRESHOLD:
            retired["silent"] = e - repair_at
        tag = "  <- repair injected" if e == repair_at else ""
        print(f"    {e:>5} | {cw:>13.3f} m | {cs:>14.3f} m | {cc:>11.3f} m{tag}")

    print()
    print(f"    works-fixed retired  {retired['works']} events after repair"
          f" ({'PASS' if retired['works'] is not None else 'FAIL'})")
    print(f"    silent-fixed retired {retired['silent']} events after repair"
          f" ({'PASS' if retired['silent'] is not None else 'FAIL'})")
    print(f"    control still flooded: {corrected(control) >= RETIRE_THRESHOLD} "
          f"(depth {corrected(control):.3f} m)")
    control_depth = corrected(control)
    eng.reset_all()
    return {
        "retired_works_after": retired["works"],
        "retired_silent_after": retired["silent"],
        "control_depth": control_depth,
        "trajectory": traj,
        "repair_at": repair_at,
        "threshold": RETIRE_THRESHOLD,
    }


if __name__ == "__main__":
    r = run_road_fixed_test()
    ok = (r["retired_works_after"] is not None and r["retired_silent_after"] is not None
          and r["control_depth"] >= RETIRE_THRESHOLD)
    print(f"\nPhase 7 {'OK' if ok else 'INCOMPLETE'}. Stale flood predictions retire "
          "automatically after a repair; the control stays flooded.")
    sys.exit(0 if ok else 1)
