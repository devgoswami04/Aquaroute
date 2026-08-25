# AquaRoute — Architecture

Predictive, road-segment-level urban-flood forecasting and vehicle-aware safe
routing for cities without digitised storm-water drainage data. Study corridor:
Chennai — Velachery / Pallikaranai / Tambaram.

Two layers, per the brief: **Predict** a per-segment water-depth-vs-time curve
(onset, peak, clearance) up to 24 h ahead; **Correct** it with a closed-loop
self-calibration layer that retires stale predictions automatically. Then convert
forecasts into vehicle-class-aware routing.

## Data flow

```
Open-Meteo (rain) ─┐
OSMnx (roads) ─────┤ INGEST (M1) ──▶ FEATURES (M2) ──▶ LABELS (M3) ──▶ MODEL (M4)
AWS Terrain (DEM) ─┘  rainfall grid    pysheds hydrology   SAR/synthetic   baseline +
GEE S1 SAR ────────┘  road graph       ~100 m segments     flood labels    Flood Response
                                        per-seg features                    Function (FRF)
                                                                                │
   sensors / traffic / works ─▶ CALIBRATION (M5) ◀── residuals vs observations │
   (synthetic feeds, real       change-point (ruptures) + online update (EWMA) │
    JSON contract)              → per-segment α retires stale predictions ──────┤
                                                                                ▼
                                DECISION LAYER (M6–M8)
                                routing (time-expanded, vehicle thresholds)
                                FastAPI  ·  React + Leaflet UI  ·  civic dashboard
```

## Modules (`backend/aquaroute/`)

| Module | Package | Responsibility |
|---|---|---|
| M1 Ingestion | `ingestion/` | rainfall (Open-Meteo), road graph (OSMnx), DEM (OpenTopography → AWS Terrain fallback), SAR (GEE) |
| M2 Features | `features/` | pysheds DEM conditioning (fill, flow dir/accum, TWI, depression), ~100 m segmentation, per-segment feature vectors |
| M3 Labels | `labels/` | **real Sentinel-1 SAR** flood masks (Planetary Computer, change detection) → per-segment labels; synthetic fallback where no SAR scene covers the event |
| M4 Model | `model/` | RF/XGBoost baseline + eval harness; **Flood Response Function** (LSTM/TCN + PyG GNN → depth-vs-time) |
| M5 Calibration | `calibration/` | residuals, change-point (ruptures), online update, public-works reset, traffic-as-non-flood |
| M6 Routing | `routing/` | time-expanded vehicle-aware safe routing, ORS avoid fallback |
| M7 API | `api/` | FastAPI endpoints |
| M8 Civic | `civic/` | decision-support summaries for the dashboard |
| — Synthetic | `synthetic/` | sensor/traffic/works feed generators + synthetic flood masks |
| — DB | `db/` | PostGIS models (SQLAlchemy + GeoAlchemy2) + GeoParquet stores |

## The novel core — Flood Response Function

Factored into **extent** (grounded in real observations) × **timing** (the FRF):

- **Extent** — `model/propensity.py`. An XGBoost classifier maps per-segment
  terrain features → flood probability, trained on **real Sentinel-1 SAR** labels.
  Held out (train one real event → predict the other) it reaches ROC-AUC ≈ 0.78, so
  it *generalises* — real SAR predicting real SAR, no circularity.
- **Timing** — `model/frf.py`. A **temporal encoder** (LSTM/TCN) over the rainfall
  hyetograph + a **PyG GNN** (GraphSAGE/GAT) over the ~121 k-node / ~900 k-edge
  segment graph → an MLP decoder emitting a **normalised depth-vs-time shape**
  (sigmoid, dense target — trains stably where a sparse absolute-depth target
  collapses). Trained on many synthetic storms so the encoder learns the
  rainfall-shape→timing mapping.

At inference: **depth = MAX_DEPTH · propensity · shape**; `derive_events` extracts
onset/peak/clearance. On the real Sentinel-1 events the model ranks flood risk at
ROC-AUC ≈ 0.93 and predicts depth (RMSE ≈ 0.02 m) and timing (onset MAE 3–5 h,
clearance 0.6–2.5 h). GAT ≈ GraphSAGE.

## The review fix — closed-loop self-calibration

`calibration/engine.py`. A per-segment multiplicative correction `α` sits on the
**frozen** FRF (`corrected = α · frf`). Each cycle: residual = predicted − observed;
a `ruptures` change-point on the residual history (or a completed public-works
event) boosts the online learning rate so a repaired road's flood prediction
retires within ~1 event. It is **never** "retrain on all data" — the reaction is
per-segment, residual- and change-point-driven (brief §11).

## Key engineering decisions

- **Keyless DEM** via AWS Terrain Tiles (OpenTopography needs a key; Open-Meteo
  elevation rate-limits on bulk points). 30 m OpenTopography is a drop-in upgrade.
- **GeoParquet** segment store (9 MB, ~1 s read) vs 68 MB GeoJSON; PostGIS optional
  with a fast socket pre-check so a dead DB never blocks the API.
- **Storm-driven FRF targets** + depth-weighted loss + a low-rainfall damping gate,
  so the model is calibrated from a dry live forecast to a 300 mm cyclone.
- Everything config-driven (`config.yaml` / `.env`); no hard-coded secrets, bbox or dates.

## Runtime

- **Backend**: FastAPI (`uvicorn aquaroute.api.main:app --app-dir backend`).
  Cold cost is paid once per process: `/predict` ~20 s (Torch + model + 121 k
  forward), `/route` ~45 s (builds the routing graph); cached per scenario after.
- **Frontend**: React + Vite + Leaflet + Recharts, proxying `/api` to the backend.
- **DB**: Postgres 16 + PostGIS via docker-compose (optional).

See [EVALUATION.md](EVALUATION.md) for §8 results and figures.
