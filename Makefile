# AquaRoute — developer entrypoints. Cross-platform-ish; Windows users can also
# run the underlying commands directly (see README).
.PHONY: help setup up down logs api ingest test frontend-install frontend fmt

help:
	@echo "AquaRoute targets:"
	@echo "  setup             - create venv & install backend deps"
	@echo "  up                - start Postgres/PostGIS (docker compose up -d)"
	@echo "  down              - stop Postgres/PostGIS"
	@echo "  logs              - tail db logs"
	@echo "  api               - run FastAPI (uvicorn) on :8000"
	@echo "  ingest            - run Phase 1 ingestion demo (rainfall + roads + DEM)"
	@echo "  features          - run Phase 2 pipeline (DEM hydrology + segments -> store)"
	@echo "  labels            - run Phase 3 pipeline (SAR/synthetic flood labels per event)"
	@echo "  baseline          - run Phase 4 pipeline (RF/XGBoost baseline + LOEO eval)"
	@echo "  frf               - run Phase 5 pipeline (Flood Response Function train + eval)"
	@echo "  calibrate         - run Phase 7 road-fixed self-calibration test"
	@echo "  route-demo        - run Phase 8 vehicle-aware routing demo"
	@echo "  eval              - run Phase 10 evaluation report (§8 figures + docs/EVALUATION.md)"
	@echo "  test              - run pytest"
	@echo "  frontend-install  - npm install in frontend/"
	@echo "  frontend          - run Vite dev server on :5173"

setup:
	python -m venv .venv
	./.venv/Scripts/python -m pip install --upgrade pip
	./.venv/Scripts/pip install -r backend/requirements.txt

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f db

api:
	uvicorn aquaroute.api.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

ingest:
	python -m aquaroute.ingestion.demo

features:
	python -m aquaroute.features.pipeline

labels:
	python -m aquaroute.labels.pipeline

baseline:
	python -m aquaroute.model.pipeline

frf:
	python -m aquaroute.model.frf_pipeline

calibrate:
	python -m aquaroute.calibration.roadfixed

route-demo:
	python -m aquaroute.routing.demo

eval:
	python -m aquaroute.eval.report

test:
	pytest -q

frontend-install:
	cd frontend && npm install

frontend:
	cd frontend && npm run dev
