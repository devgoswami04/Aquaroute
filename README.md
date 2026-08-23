# AquaRoute

Predictive, road-segment-level **urban-flood forecasting** and **vehicle-aware safe
routing** for cities without digitised storm-water drainage data. Study corridor:
**Chennai — Velachery / Pallikaranai / Tambaram**.

Two layers: **Predict** a per-segment water-depth-vs-time curve (onset, peak,
clearance) up to 24 h ahead, and **Correct** it with a closed-loop
self-calibration layer that retires stale flood predictions automatically. Then
convert forecasts into vehicle-class-aware routing (a 15 cm road stops a
two-wheeler but not a bus).

This repo is being built **phase by phase** — see the build brief and §10 there.
**Current state: complete — Phases 0–10 all done.** Ingestion → hydrology &
features → flood labels → baseline → the novel Flood Response Function → live
forecast API & map → closed-loop self-calibration → vehicle-aware routing → civic
dashboard → full evaluation. See **[docs/EVALUATION.md](docs/EVALUATION.md)** for
the §8 results and **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the design.

## Results at a glance (§8)

| Model | Event F1 (LOEO) | Depth RMSE | Onset MAE | Clearance MAE |
|---|---|---|---|---|
| XGBoost baseline | 0.74 | — (no depth) | — (no timing) | — (no timing) |
| RandomForest baseline | 0.80 | — | — | — |
| **Flood Response Function** | 0.69 | **0.07 m** | **1.0 h** | **3.8 h** |

The FRF matches the baseline's classification while adding the depth-vs-time
timing the baseline structurally cannot produce. Self-calibration retires a
repaired road's flood prediction within **~1 event**; safe routing cuts flood
exposure from ~8 km (shortest) to ~1 km for a two-wheeler, and a bus finds a
fully-passable route where a two-wheeler cannot. Regenerate everything with:

```bash
python -m aquaroute.eval.report
```

## Repository layout

```
├── docker-compose.yml     # Postgres 16 + PostGIS
├── Makefile               # setup / up / api / ingest / test / frontend
├── config.yaml            # bbox, events, thresholds, hyperparameters
├── .env.example           # API keys & DB creds (copy to .env)
├── backend/
│   ├── requirements.txt
│   └── aquaroute/
│       ├── config.py                # single config/secret loader
│       ├── api/main.py              # FastAPI (Phase 0: /health)
│       ├── ingestion/               # Module 1 — rainfall, roads, DEM (Phase 1)
│       └── features/ labels/ model/ calibration/ routing/ db/ synthetic/  # later phases
├── frontend/              # React + Vite + Leaflet (Phase 0 shell)
├── tests/                 # pytest (Phase 1 ingestion tests)
├── data/                  # cached DEM / graphs (gitignored)
└── notebooks/
```

## Prerequisites

- Python 3.11+ (tested on 3.13), Node 18+, Docker (for PostGIS).

## Setup & run

**1. Python environment**

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Git Bash:            source .venv/Scripts/activate
pip install -r backend/requirements.txt
```

**2. Config**

```bash
cp .env.example .env
```

Edit `.env` if you have free keys (OpenTopography for the DEM; others optional).
Rainfall and the road graph need **no key**.

**3. Database (PostGIS)**

```bash
docker compose up -d
```

**4. Backend API** (health check for Phase 0)

```bash
uvicorn aquaroute.api.main:app --app-dir backend --reload --port 8000
```

Verify: <http://localhost:8000/health> returns `{"status":"ok",...}`; interactive
docs at <http://localhost:8000/docs>.

**5. Phase 1 ingestion demo** — prints a 24 h rainfall series + road-graph segment count:

```bash
python -m aquaroute.ingestion.demo
```

(Run from repo root with `backend/` on `PYTHONPATH`, or `make ingest`.)

**5b. Phase 2 feature pipeline** — DEM hydrology → ~100 m segments → features → store:

```bash
python -m aquaroute.features.pipeline
```

Writes `data/segments.parquet` (GeoParquet, ~9 MB; and PostGIS if the DB is up).
Needs no key — falls back to keyless **AWS Terrain Tiles** (~18 m DEM) when
`OPENTOPOGRAPHY_API_KEY` is unset (add the free key for the 30 m SRTM DEM). Then
`GET /segments` serves filtered GeoJSON and the frontend map renders it.

`GET /segments` accepts `?classes=`, `?bbox=west,south,east,north`, `?limit=`.
The map requests major road classes by default (~15.6k of 121k segments) so
Leaflet stays responsive; residential streets are in the store for routing.

**5c. Phase 3 flood labels** — per-event ground truth from SAR (or synthetic):

```bash
python -m aquaroute.labels.pipeline
```

Labels every configured event (2015 / 2021 / 2023). Uses Sentinel-1 SAR via Google
Earth Engine when `EE_PROJECT` is set (run `earthengine authenticate` once);
otherwise a **rainfall-driven synthetic** water mask (real Open-Meteo event totals
set the flood extent) — identical GeoTIFF contract, so the SAR path is a drop-in.
Writes `data/labels/<event>.parquet`. Then pick an event in the UI dropdown to
overlay flooded segments (blue) on the map, or call `GET /labels?event=<name>`.
`GET /events` lists events and whether they're labelled.

**5d. Phase 4 baseline** — susceptibility model + leave-one-event-out evaluation:

```bash
python -m aquaroute.model.pipeline
```

Trains RF/XGBoost on `assemble_training_set()` and evaluates by holding out each
event in turn (§8). Prints the metrics table (the comparison bar for Phase 5) and
saves `data/models/baseline_xgb.joblib` + `baseline_metrics.json`. Example run:
XGBoost LOEO mean F1 ≈ 0.74, RF ≈ 0.80 — well below the optimistic random-split
numbers, which is the point: LOEO measures real cross-event generalization.

**5e. Phase 5 Flood Response Function** — the novel model (needs PyTorch + PyG):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
python -m aquaroute.model.frf_pipeline
```

A **temporal encoder** (LSTM/TCN) over the rainfall hyetograph + a **PyG GNN**
(GraphSAGE/GAT) over the ~121k-node segment graph → a per-segment **depth-vs-time
curve** (`predict_curve` / `derive_events` give onset/peak/clearance). Trained on
synthetic storms (so the encoder learns rainfall-shape→timing) and validated on the
three real historical storms. Example run: **depth RMSE ≈ 0.07 m, onset MAE ≈ 1 h,
clearance MAE ≈ 4 h, event F1 ≈ 0.72** — on par with the baseline's F1 *and* adding
the onset/clearance timing the baseline structurally cannot produce (§8). Saves
`data/models/frf.pt`.

**Phase 6 live forecast** — no extra step: once `frf.pt` exists, the API serves it.

* `GET /predict?scenario=live&classes=…` — per-segment peak depth + onset/peak/
  clearance for the current Open-Meteo forecast, as GeoJSON (`scenario=` also
  accepts a historical event slug like `2021_chennai_floods` to *replay* that storm).
* `GET /segment/{id}/curve?scenario=…` — the predicted depth-vs-time series + the
  driving hyetograph, for the chart.

In the UI: switch **Layer → Forecast (FRF)**, pick **Live** or a replay scenario;
roads recolour by predicted peak depth, and clicking a segment shows its
depth-vs-time chart with onset/peak/clearance markers.

**Phase 7 self-calibration** — the closed loop that keeps the model honest:

```bash
python -m aquaroute.calibration.roadfixed   # the "road-fixed" test
```

A per-segment multiplicative correction `alpha` sits on top of the (frozen) FRF.
Each cycle compares predicted vs observed depth (from synthetic
sensor/traffic/works feeds with a real-feed JSON contract), updates `alpha` online
(EWMA), and reacts to **structural change** — a `ruptures` change-point or a
completed public-works event — by boosting the learning rate so **stale flood
predictions retire within a few events**. The road-fixed test shows a works-fixed
segment retiring 1 event after repair, a silently-fixed one caught by the
change-point detector, and a control staying flooded. Live endpoints:
`POST /ingest/sensor`, `POST /ingest/works`, `POST /report`, `POST /calibrate/run`,
`GET /calibrate/status`; retired segments then read shallower in `/predict`.

**Phase 8 vehicle-aware routing** — different safe routes per vehicle class:

```bash
python -m aquaroute.routing.demo
```

`VEHICLE_THRESHOLDS` (two-wheeler 0.15 m … bus 0.60 m) drive a **time-expanded**
routing graph: an edge is impassable when the water at the vehicle's arrival hour
exceeds its threshold. `safe_route` runs a time-dependent Dijkstra minimising
**routing regret** (travel time + flood-exposure penalty), with an ORS avoid-
polygons fallback. On the 2021 replay at peak rain, a bus finds a fully-passable
22 km route while a two-wheeler detours 28 km. `POST /route` (body
`{origin,dest,depart_time,vehicle,scenario}`) returns safe vs shortest + a
plain-language advisory; `GET /vehicles` lists the thresholds. In the UI, the
**Routing** panel: pick vehicle + scenario + depart hour, click the map to set
origin/destination, and it draws the safe (blue) vs shortest (dashed) routes.

**Phase 9 civic dashboard** — `GET /civic/summary` and the **Dashboard** view
(toggle top-left). Three things a planner needs: a **chronic-flood segment
ranking** (which roads flood in the most historical events → the evidence base for
drainage works), **predicted-vs-observed** agreement (FRF vs SAR/report labels, per
event), and **self-calibration status** ("self-corrected N segments"). This is the
civic-decision-support framing, not a navigation clone.

**6. Frontend shell**

```bash
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173> — it displays the backend `/health` payload.

## Tests

```bash
pytest -q
# offline (skip API calls):
AQUAROUTE_SKIP_NETWORK=1 pytest -q
```

## Data sources (all free)

Rainfall — Open-Meteo (keyless). Road graph — OpenStreetMap via OSMnx (keyless).
DEM — OpenTopography (free key). SAR flood labels — Sentinel-1 via Google Earth
Engine (free account, Phase 3). See the build brief §4 for the full table.

## What each phase adds

Phase 0 scaffold · **Phase 1 ingestion (done)** · **Phase 2 DEM hydrology + road
segmentation + features + map (done)** · **Phase 3 flood labels + overlay (done)** ·
**Phase 4 baseline + LOEO eval (done)** · **Phase 5 Flood Response Function (done)** ·
**Phase 6 live predict API + forecast map + depth chart (done)** ·
**Phase 7 self-calibration loop (done)** · **Phase 8 vehicle-aware routing (done)** ·
**Phase 9 civic dashboard (done)** · **Phase 10 evaluation & docs (done)**.
