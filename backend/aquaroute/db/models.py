"""PostGIS data model (brief §7).

Declared with SQLAlchemy 2.0 + GeoAlchemy2. The ``segments`` table is populated in
Phase 2; the remaining tables (events, rainfall, flood_labels, predictions,
observations, calibration_log, citizen_reports) are defined here so later phases
write against a stable schema.
"""
from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Segment(Base):
    __tablename__ = "segments"

    segment_id: Mapped[str] = mapped_column(String, primary_key=True)
    u: Mapped[int | None] = mapped_column(Integer, nullable=True)
    v: Mapped[int | None] = mapped_column(Integer, nullable=True)
    key: Mapped[int] = mapped_column(Integer, default=0)
    road_class: Mapped[str | None] = mapped_column(String, nullable=True)
    is_underpass: Mapped[bool] = mapped_column(Boolean, default=False)
    length_m: Mapped[float] = mapped_column(Float)
    elevation: Mapped[float | None] = mapped_column(Float, nullable=True)
    slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    twi: Mapped[float | None] = mapped_column(Float, nullable=True)
    depression_depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    upstream_area: Mapped[float | None] = mapped_column(Float, nullable=True)
    imperviousness: Mapped[float | None] = mapped_column(Float, nullable=True)
    susceptibility: Mapped[float | None] = mapped_column(Float, nullable=True)
    geom: Mapped[object] = mapped_column(Geometry("LINESTRING", srid=4326))


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    start_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    end_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)


class Rainfall(Base):
    __tablename__ = "rainfall"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"), nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    cell_geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326))
    precip_mm: Mapped[float] = mapped_column(Float)


class FloodLabel(Base):
    __tablename__ = "flood_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"), nullable=True)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.segment_id"))
    flooded: Mapped[bool] = mapped_column(Boolean)
    depth_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.segment_id"))
    ts: Mapped[datetime] = mapped_column(DateTime)
    depth_pred: Mapped[float | None] = mapped_column(Float, nullable=True)
    onset: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    peak: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    clearance: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.segment_id"))
    ts: Mapped[datetime] = mapped_column(DateTime)
    depth_obs: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String)  # sensor|traffic|report|works


class CalibrationLog(Base):
    __tablename__ = "calibration_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.segment_id"))
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"), nullable=True)
    residual: Mapped[float | None] = mapped_column(Float, nullable=True)
    change_point: Mapped[bool] = mapped_column(Boolean, default=False)
    action: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class CitizenReport(Base):
    __tablename__ = "citizen_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    segment_id: Mapped[str | None] = mapped_column(
        ForeignKey("segments.segment_id"), nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String)  # flooded|clear
    depth_est: Mapped[float | None] = mapped_column(Float, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
