"""AquaRoute FastAPI app.

Phase 0: health check + config echo, so the stack is verifiably runnable before
any data work. Later phases mount the /segments, /predict, /route, /report,
/ingest, /calibrate and /civic routers (brief §6, Module 7).
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aquaroute import __version__
from aquaroute.config import get_config

# Warm-up state, so /health can report readiness while caches load.
_WARM = {"segments": False, "forecast": False, "routing": False}


def _warm_caches() -> None:
    """Pre-load the heavy caches (segments, FRF model, routing graph) in the
    background at startup, so the first user request isn't a 15–45 s cold start.
    Each step imports lazily and swallows errors (e.g. no FRF model yet)."""
    try:
        from aquaroute.db.segments_store import load_segments_gdf
        load_segments_gdf()
        _WARM["segments"] = True
    except Exception:
        pass
    try:
        from aquaroute.model.predictor import get_predictor
        p = get_predictor()
        p.predict_current("live")
        p.predict_current("2021_chennai_floods")   # the vivid replay demo
        _WARM["forecast"] = True
    except Exception:
        pass
    try:
        from aquaroute.routing import get_router
        get_router("2021_chennai_floods")           # routing-panel default scenario
        _WARM["routing"] = True
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_caches, name="aquaroute-warm", daemon=True).start()
    yield


class FeedRecords(BaseModel):
    records: list[dict]                    # sensor/traffic feed records (carry 'source')


class WorksEvent(BaseModel):
    segment_ids: list[str]
    work_type: str = "desilting"


class CitizenReport(BaseModel):
    segment_id: str
    status: str                            # flooded | clear
    depth_est: float | None = None
    note: str = ""


class CalibrateRequest(BaseModel):
    scenario: str = "live"


class RouteRequest(BaseModel):
    origin: list[float]                    # [lon, lat]
    dest: list[float]                      # [lon, lat]
    depart_time: int = 0                   # hour offset into the forecast (0–23)
    vehicle: str = "car"
    scenario: str = "live"

app = FastAPI(
    title="AquaRoute API",
    version=__version__,
    summary="Predictive urban-flood forecasting + vehicle-aware safe routing.",
    lifespan=lifespan,
)

# Allow the Vite dev server to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness probe used by `make up` verification (brief Phase 0)."""
    cfg = get_config()
    return {
        "status": "ok",
        "service": "aquaroute",
        "version": __version__,
        "bbox": {
            "south": cfg.bbox.south,
            "west": cfg.bbox.west,
            "north": cfg.bbox.north,
            "east": cfg.bbox.east,
        },
        "events": [e["name"] for e in cfg.events],
        "warm": dict(_WARM),
    }


@app.get("/segments")
def segments(classes: str | None = None, bbox: str | None = None,
             limit: int | None = None) -> dict:
    """Segment geometries + current risk state as a GeoJSON FeatureCollection.

    Query params (all optional):
      * ``classes`` — comma-separated road_class filter (e.g. ``primary,secondary``)
      * ``bbox``    — ``west,south,east,north`` spatial filter
      * ``limit``   — cap the number of features

    Until the model lands (Phase 5/6), the ``susceptibility`` property is a
    terrain-derived heuristic used only for map colouring. Reads PostGIS if the DB
    is up, otherwise the ``data/segments.geojson`` cache from the Phase 2 pipeline.
    """
    from aquaroute.db.segments_store import load_segments_geojson

    class_list = [c.strip() for c in classes.split(",")] if classes else None
    bbox_tuple = None
    if bbox:
        try:
            w, s, e, n = (float(x) for x in bbox.split(","))
            bbox_tuple = (w, s, e, n)
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox must be 'west,south,east,north'")

    try:
        return load_segments_geojson(classes=class_list, bbox=bbox_tuple, limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e) + " (run the Phase 2 feature pipeline first)",
        )


@app.get("/predict")
def predict(classes: str | None = None, bbox: str | None = None,
            limit: int | None = None, only_flooded: bool = False,
            scenario: str = "live", refresh: bool = False) -> dict:
    """Per-segment forecast (peak depth + onset/peak/clearance) as GeoJSON.

    Runs the trained Flood Response Function on a rainfall hyetograph selected by
    ``scenario``: ``live`` (the current Open-Meteo forecast, default) or a
    historical event slug (e.g. ``2021_chennai_floods``) to replay that storm.
    Results are cached per scenario in-process; pass ``refresh=true`` to recompute.
    """
    from aquaroute.model.predictor import get_predictor

    class_list = [c.strip() for c in classes.split(",")] if classes else None
    bbox_tuple = None
    if bbox:
        try:
            w, s, e, n = (float(x) for x in bbox.split(","))
            bbox_tuple = (w, s, e, n)
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox must be 'west,south,east,north'")
    try:
        return get_predictor().predict_geojson(
            classes=class_list, bbox=bbox_tuple, limit=limit,
            only_flooded=only_flooded, scenario=scenario, refresh=refresh)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/segment/{segment_id}/curve")
def segment_curve(segment_id: str, scenario: str = "live") -> dict:
    """Predicted depth-vs-time curve + hyetograph for one segment (for the chart)."""
    from aquaroute.model.predictor import get_predictor

    try:
        return get_predictor().segment_curve(segment_id, scenario=scenario)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown segment_id: {segment_id}")


@app.get("/events")
def events() -> list:
    """Configured historical events + whether flood labels have been built."""
    from aquaroute.labels.store import dominant_source, list_labelled_events, slug

    cfg = get_config()
    labelled = set(list_labelled_events())
    out = []
    for e in cfg.events:
        is_labelled = slug(e["name"]) in labelled
        out.append({
            "name": e["name"], "start": e["start"], "end": e["end"],
            "labelled": is_labelled,
            "label_source": dominant_source(e["name"]) if is_labelled else None,
        })
    return out


@app.get("/labels")
def labels(event: str, classes: str | None = None, only_flooded: bool = True) -> dict:
    """Per-event flood labels joined to segment geometry (GeoJSON overlay).

    ``event`` may be the display name or its slug. ``only_flooded`` (default true)
    returns just the flooded segments for a clean map overlay.
    """
    from aquaroute.labels.store import load_labels_geojson, slug

    class_list = [c.strip() for c in classes.split(",")] if classes else None
    try:
        return load_labels_geojson(slug(event), classes=class_list, only_flooded=only_flooded)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Module 5: self-calibration feeds & loop ---

@app.post("/ingest/sensor")
def ingest_sensor(body: FeedRecords) -> dict:
    """Buffer sensor/traffic feed records (real or synthetic) for calibration."""
    from aquaroute.calibration.engine import get_engine
    get_engine().ingest(body.records)
    return {"buffered": len(body.records)}


@app.post("/ingest/works")
def ingest_works(body: WorksEvent) -> dict:
    """Register completed public works → reset & boost those segments' calibration."""
    from aquaroute.calibration.engine import get_engine
    eng = get_engine()
    acts = [eng.apply_public_works_reset(s, body.work_type) for s in body.segment_ids]
    eng.save()
    return {"reset": acts}


@app.post("/report")
def report(body: CitizenReport) -> dict:
    """Citizen flood/clear report → observation feeding the calibration loop."""
    from aquaroute.calibration.engine import get_engine
    rec = {"segment_id": body.segment_id, "status": body.status,
           "depth_est": body.depth_est, "note": body.note, "source": "report", "ts": None}
    get_engine().ingest([rec])
    return {"ok": True, "segment_id": body.segment_id}


@app.post("/calibrate/run")
def calibrate_run(body: CalibrateRequest) -> dict:
    """Run one calibration cycle: FRF predictions vs buffered observations →
    residuals → change-point → online update. Retired segments show shallower in
    subsequent /predict calls."""
    from aquaroute.calibration.engine import get_engine
    from aquaroute.model.predictor import get_predictor

    preds = get_predictor().raw_peaks(body.scenario)
    log = get_engine().run_cycle(preds, observations=None)
    return {
        "segments_updated": int(len(log)),
        "change_points": int(log["change_point"].sum()) if len(log) else 0,
        "summary": get_engine().summary(),
    }


@app.get("/calibrate/status")
def calibrate_status() -> dict:
    """Current calibration state (tracked/retired segment counts, mean alpha)."""
    from aquaroute.calibration.engine import get_engine
    return get_engine().summary()


# --- Module 6: vehicle-aware routing ---

@app.get("/vehicles")
def vehicles() -> dict:
    """Vehicle classes and their flood depth thresholds (m)."""
    from aquaroute.routing import vehicle_thresholds
    return vehicle_thresholds()


@app.post("/route")
def route(body: RouteRequest) -> dict:
    """Vehicle-aware safe route vs shortest route for the forecast, with advisory.

    ``origin``/``dest`` are ``[lon, lat]``; ``depart_time`` is an hour (0–23) into
    the forecast; ``vehicle`` ∈ two_wheeler|auto|car|bus.
    """
    from aquaroute.routing import safe_route

    try:
        return safe_route(tuple(body.origin), tuple(body.dest),
                          body.depart_time, body.vehicle, body.scenario)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# --- Module 8: civic decision dashboard ---

@app.get("/civic/summary")
def civic_summary(top_n: int = 20, refresh: bool = False) -> dict:
    """Decision-support summary: chronic-flood segment ranking, predicted-vs-
    observed agreement, and self-calibration status."""
    from aquaroute.civic import build_civic_summary
    return build_civic_summary(top_n=top_n, refresh=refresh)


@app.get("/")
def root() -> dict:
    return {"message": "AquaRoute API. See /docs for the OpenAPI UI, /health for status."}
