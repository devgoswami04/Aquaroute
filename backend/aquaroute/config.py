"""Central config loader.

Loads non-secret parameters from ``config.yaml`` (repo root) and secrets from the
environment / ``.env``. Nothing else in the codebase should read env vars or the
YAML directly — import ``get_config()`` / ``get_settings()`` instead. This keeps
the "config-driven, no hard-coded secrets" rule (brief §2.4) enforceable.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    """Repo root = two levels up from this file (backend/aquaroute/config.py)."""
    return Path(__file__).resolve().parents[2]


# Load .env once at import time so os.environ is populated for Settings below.
load_dotenv(_repo_root() / ".env")


class Settings(BaseSettings):
    """Secrets & connection strings, sourced from the environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(
        default="postgresql+psycopg://aquaroute:aquaroute@localhost:5432/aquaroute",
        alias="DATABASE_URL",
    )
    opentopography_api_key: str = Field(default="", alias="OPENTOPOGRAPHY_API_KEY")
    ee_project: str = Field(default="", alias="EE_PROJECT")
    tomtom_api_key: str = Field(default="", alias="TOMTOM_API_KEY")
    ors_api_key: str = Field(default="", alias="ORS_API_KEY")
    nominatim_user_agent: str = Field(default="aquaroute-dev", alias="NOMINATIM_USER_AGENT")


class BBox:
    """Bounding box helper. Order is explicit to avoid lat/lon swaps."""

    def __init__(self, south: float, west: float, north: float, east: float):
        self.south, self.west, self.north, self.east = south, west, north, east

    # Common orderings various libraries expect.
    def as_osmnx(self) -> tuple[float, float, float, float]:
        """OSMnx bbox order: (north, south, east, west)."""
        return (self.north, self.south, self.east, self.west)

    def as_west_south_east_north(self) -> tuple[float, float, float, float]:
        """GDAL / rasterio / OpenTopography order: (west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)

    def center(self) -> tuple[float, float]:
        """(lat, lon) centre."""
        return ((self.south + self.north) / 2, (self.west + self.east) / 2)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"BBox(S={self.south}, W={self.west}, N={self.north}, E={self.east})"


class Config:
    """Typed view over config.yaml plus the resolved bbox and secrets."""

    def __init__(self, raw: dict[str, Any], settings: Settings):
        self._raw = raw
        self.settings = settings
        b = raw["bbox"]
        self.bbox = BBox(b["south"], b["west"], b["north"], b["east"])
        self.place: str = raw.get("place", "")
        self.crs: str = raw["project"]["crs"]
        self.metric_crs: str = raw["project"]["metric_crs"]
        self.events: list[dict[str, str]] = raw.get("events", [])
        self.ingestion: dict[str, Any] = raw.get("ingestion", {})
        self.features: dict[str, Any] = raw.get("features", {})
        self.vehicle_thresholds: dict[str, float] = raw.get("vehicle_thresholds", {})
        self.model: dict[str, Any] = raw.get("model", {})

    @property
    def cache_dir(self) -> Path:
        d = _repo_root() / self.ingestion.get("cache_dir", "data")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup: config.get('model', 'frf', 'hidden_dim')."""
        node: Any = self._raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    path = _repo_root() / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw, get_settings())
