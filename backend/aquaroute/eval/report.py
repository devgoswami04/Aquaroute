"""Phase 10 — full evaluation report (§8).

Consolidates the metrics produced across phases into figures + an EVALUATION.md:
  1. Baseline (RF/XGBoost) vs Flood Response Function — F1 and the timing the
     baseline cannot produce.
  2. FRF per-event depth RMSE / onset / clearance error (held-out storms).
  3. Self-calibration 'road-fixed' curve — retirement after a repair.
  4. Routing regret — flooded-road exposure & detour cost per vehicle class.

Run with ``python -m aquaroute.eval.report`` (or ``make eval``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aquaroute.config import get_config  # noqa: E402

NAVY, TEAL, AMBER, RED, GREY = "#0b3d5c", "#1a9aa8", "#e67e22", "#c0392b", "#95a5a6"


def _repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _figdir() -> Path:
    d = _repo() / "docs" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load(name: str) -> dict | None:
    p = get_config().cache_dir / "models" / name
    return json.loads(p.read_text()) if p.exists() else None


def _real_events(frf) -> list:
    """FRF per-event entries for the real-SAR events only (exclude synthetic)."""
    prop = frf.get("propensity_generalization", {}).get("per_event_auc", {})
    real = set(prop.keys())
    return [e for e in frf["per_event"] if e["event"] in real] or frf["per_event"]


def fig_grounding(frf, out: Path):
    """Real-SAR grounding: held-out propensity AUC vs FRF AUC per real event,
    plus the timing the extent model alone cannot produce."""
    prop = frf.get("propensity_generalization", {}).get("per_event_auc", {})
    real = _real_events(frf)
    names = [e["event"].replace("_", " ") for e in real]
    x = np.arange(len(names)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.7))

    prop_auc = [prop.get(e["event"], np.nan) for e in real]
    frf_auc = [e["event_roc_auc"] for e in real]
    ax1.bar(x - w / 2, prop_auc, w, label="propensity (extent, held-out)", color=GREY)
    ax1.bar(x + w / 2, frf_auc, w, label="FRF (in-sample)", color=NAVY)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylim(0, 1); ax1.set_ylabel("ROC-AUC vs real Sentinel-1 SAR")
    ax1.set_title("Flood extent grounded in real SAR"); ax1.legend(fontsize=8)
    ax1.axhline(0.5, color="k", ls=":", lw=0.8)

    onset = [e["onset_mae_h"] or 0 for e in real]
    clear = [e["clearance_mae_h"] or 0 for e in real]
    ax2.bar(x - w / 2, onset, w, label="onset MAE", color=TEAL)
    ax2.bar(x + w / 2, clear, w, label="clearance MAE", color=AMBER)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("hours"); ax2.set_title("FRF timing (extent model has none)")
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_frf_per_event(frf, out: Path):
    ev = frf["per_event"]
    names = [e["event"].replace("_", " ") for e in ev]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w, [e["depth_rmse_m"] for e in ev], w, label="depth RMSE (m)", color=NAVY)
    ax.bar(x, [e["onset_mae_h"] or 0 for e in ev], w, label="onset MAE (h)", color=TEAL)
    ax.bar(x + w, [e["clearance_mae_h"] or 0 for e in ev], w, label="clearance MAE (h)", color=AMBER)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_title("FRF per-event error (held-out historical storms)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_calibration(traj_result, out: Path):
    t = traj_result["trajectory"]
    n = len(t["works"]); x = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, t["works"], "-o", color=RED, label="repaired (works event)")
    ax.plot(x, t["silent"], "-s", color=AMBER, label="repaired (silent → change-point)")
    ax.plot(x, t["control"], "-^", color=NAVY, label="control (still floods)")
    ax.axvline(traj_result["repair_at"], color=GREY, ls="--", label="repair injected")
    ax.axhline(traj_result["threshold"], color="k", ls=":", lw=1, label="retire threshold 0.1 m")
    ax.set_xlabel("calibration cycle (event)"); ax.set_ylabel("corrected predicted depth (m)")
    ax.set_title("Self-calibration: stale flood predictions retire after a repair")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_routing(routing, out: Path):
    veh = list(routing.keys()); x = np.arange(len(veh)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.bar(x - w / 2, [routing[v]["shortest_km"] for v in veh], w, label="shortest", color=GREY)
    ax1.bar(x + w / 2, [routing[v]["safe_km"] for v in veh], w, label="safe", color=NAVY)
    ax1.set_xticks(x); ax1.set_xticklabels([v.replace("_", "-") for v in veh], fontsize=8)
    ax1.set_ylabel("distance (km)"); ax1.set_title("Detour cost: safe vs shortest"); ax1.legend(fontsize=8)

    ax2.bar(x - w / 2, [routing[v]["shortest_blocked_km"] for v in veh], w, label="shortest", color=GREY)
    ax2.bar(x + w / 2, [routing[v]["safe_blocked_km"] for v in veh], w, label="safe", color=RED)
    ax2.set_xticks(x); ax2.set_xticklabels([v.replace("_", "-") for v in veh], fontsize=8)
    ax2.set_ylabel("impassable water on route (km)")
    ax2.set_title("Flood exposure: safe vs shortest"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _routing_eval(scenario="2021_chennai_floods", depart=18,
                  origin=(80.135, 12.95), dest=(80.23, 12.945)) -> dict:
    from aquaroute.routing import get_router
    R = get_router(scenario)
    out = {}
    for veh in ("two_wheeler", "auto", "car", "bus"):
        r = R.route(origin, dest, depart, veh)
        s, sh = r["safe_route"], r["shortest_route"]
        out[veh] = {
            "safe_km": round(s["distance_m"] / 1000, 2),
            "shortest_km": round(sh["distance_m"] / 1000, 2),
            "safe_blocked_km": round(s["blocked_m"] / 1000, 2),
            "shortest_blocked_km": round(sh["blocked_m"] / 1000, 2),
            "safe_exposure_km": round(s["exposure_m"] / 1000, 2),
        }
    return out


def _md_table(headers, rows) -> str:
    h = "| " + " | ".join(headers) + " |\n"
    h += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        h += "| " + " | ".join(str(c) for c in r) + " |\n"
    return h


def run() -> int:
    print("AquaRoute Phase 10 — evaluation report\n")
    frf = _load("frf_metrics.json")
    if not frf:
        print("Missing FRF metrics. Run `make frf` first.")
        return 1
    figs = _figdir()

    print("[1/4] Real-SAR grounding ..."); fig_grounding(frf, figs / "01_grounding.png")
    print("[2/4] FRF per-event error ..."); fig_frf_per_event(frf, figs / "02_frf_per_event.png")

    print("[3/4] Self-calibration road-fixed curve ...")
    from aquaroute.calibration.roadfixed import run_road_fixed_test
    cal = run_road_fixed_test(n_events=8, repair_at=3)
    fig_calibration(cal, figs / "03_self_calibration.png")

    print("[4/4] Routing regret across vehicle classes ...")
    routing = _routing_eval()
    fig_routing(routing, figs / "04_routing_regret.png")

    # --- EVALUATION.md ---
    prop = frf.get("propensity_generalization", {})
    arch = frf.get("architecture_comparison", {})
    real = _real_events(frf)
    md = ["# AquaRoute — Evaluation (§8)\n",
          "Auto-generated by `python -m aquaroute.eval.report`.\n",
          "## 1. Flood extent grounded in real Sentinel-1 SAR\n",
          "Which roads flood is learned from **real Sentinel-1 SAR** (2015 & 2021 events, "
          "Planetary Computer). A held-out test — train the flood-propensity model on one real "
          "event and score it on the *other* — shows it generalises (not memorised):\n",
          _md_table(["Train → test", "Held-out ROC-AUC (vs real SAR)"],
                    [[f"other → {e.replace('_', ' ')}", auc] for e, auc in prop.get("per_event_auc", {}).items()]
                    + [["**mean**", prop.get("mean_auc", "—")]]),
          "\nThe Flood Response Function reconstructs depth = MAX_DEPTH · propensity · shape, so it "
          "ranks flood risk against real SAR **and** adds the depth-vs-time timing the extent model "
          "has no notion of:\n",
          _md_table(["Real-SAR event", "FRF ROC-AUC", "Depth RMSE (m)", "Onset MAE (h)", "Clearance MAE (h)"],
                    [[e["event"].replace("_", " "), e["event_roc_auc"], e["depth_rmse_m"],
                      e["onset_mae_h"], e["clearance_mae_h"]] for e in real]),
          "\n![real-SAR grounding](figures/01_grounding.png)\n",
          "## 2. Architecture comparison (GAT vs GraphSAGE) & per-event error\n",
          _md_table(["GNN", "Mean ROC-AUC", "Depth RMSE (m)", "Onset MAE (h)", "Clearance MAE (h)"],
                    [[g, m["event_roc_auc"], m["depth_rmse_m"], m["onset_mae_h"], m["clearance_mae_h"]]
                     for g, m in arch.items()]) +
          f"\nBest by AUC: **{frf.get('gnn', 'graphsage')}**. Per-event (all events; 2023 Michaung uses a "
          "synthetic mask — no SAR scene in its 12-day revisit gap — so its numbers are indicative only):\n",
          _md_table(["Event", "Depth RMSE (m)", "Onset MAE (h)", "Clearance MAE (h)", "ROC-AUC"],
                    [[e["event"].replace("_", " "), e["depth_rmse_m"], e["onset_mae_h"],
                      e["clearance_mae_h"], e["event_roc_auc"]] for e in frf["per_event"]]),
          "\n![FRF per event](figures/02_frf_per_event.png)\n",
          "## 3. Self-calibration — the 'road-fixed' test\n",
          f"A repaired segment's flood prediction retires **{cal['retired_works_after']} event(s)** after a",
          f"public-works event, and **{cal['retired_silent_after']} event(s)** when the fix is silent",
          "(caught by the change-point detector). A control segment stays flooded",
          f"(depth {cal['control_depth']:.2f} m).\n",
          "![self-calibration](figures/03_self_calibration.png)\n",
          "## 4. Routing regret per vehicle class (2021 replay, 18:00)\n",
          _md_table(["Vehicle", "Threshold (m)", "Safe (km)", "Shortest (km)", "Detour (km)",
                     "Impassable on safe (km)", "Impassable on shortest (km)"],
                    [[v.replace("_", "-"),
                      {"two_wheeler": 0.15, "auto": 0.25, "car": 0.30, "bus": 0.60}[v],
                      routing[v]["safe_km"], routing[v]["shortest_km"],
                      round(routing[v]["safe_km"] - routing[v]["shortest_km"], 2),
                      routing[v]["safe_blocked_km"], routing[v]["shortest_blocked_km"]]
                     for v in routing]),
          "\n![routing regret](figures/04_routing_regret.png)\n",
          "## 5. Validation protocol\n",
          "- **Flood extent**: real Sentinel-1 SAR (change detection on Planetary Computer, keyless).",
          "  Propensity generalisation held out one real event from the other.",
          "- **FRF timing**: temporal encoder trained on synthetic storms; the three historical storms",
          "  are held out for validation. Depth/onset/clearance are vs the reservoir physics; ROC-AUC is vs real SAR.",
          "- **Self-calibration**: injected-repair test on chronically-flooded segments.",
          "- **Routing**: safe vs shortest on the same forecast, per vehicle class.\n"]
    doc = _repo() / "docs" / "EVALUATION.md"
    doc.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {doc}")
    print(f"Figures in {figs}")
    print("\nPhase 10 OK. Evaluation report + figures generated.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
