"""DEM conditioning & terrain features (Module 2).

Wraps **pysheds** for the solved hydrology (sink-fill, D8 flow direction, flow
accumulation) and adds slope, topographic wetness index (TWI) and depression
depth on top. We do not reimplement flow routing — pysheds owns that (brief §2.1).

The DEM is stored in EPSG:4326 (degrees); slope/area need metres, so cell size is
converted using the bbox-centre latitude. This is approximate but adequate at the
segment scale and keeps everything in one CRS for sampling.

Public API
----------
condition_dem(dem_tif) -> HydroLayers
compute_twi(layers) is folded into condition_dem (returns TWI in the result).
sample_layers_at(layers, lon, lat) -> dict   # per-point feature lookup
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# pysheds (<=0.4) still calls np.in1d, removed in NumPy 2.0. Shim it to np.isin so
# we can keep NumPy 2 for geopandas/rasterio. Behaviour is identical for 1-D input.
if not hasattr(np, "in1d"):
    np.in1d = np.isin  # type: ignore[attr-defined]


@dataclass
class HydroLayers:
    """Conditioned DEM and derived terrain rasters, all on the same grid."""

    dem: np.ndarray            # original elevation (m)
    filled: np.ndarray         # pit+depression filled elevation (m)
    flow_dir: np.ndarray       # D8 flow direction
    flow_acc: np.ndarray       # flow accumulation (cell counts)
    slope: np.ndarray          # local slope (radians)
    twi: np.ndarray            # topographic wetness index
    depression_depth: np.ndarray  # filled - original (m, >= 0)
    transform: object          # affine transform (raster -> lon/lat)
    cell_area_m2: float
    cell_width_m: float
    nrows: int
    ncols: int

    def rowcol(self, lon: float, lat: float) -> tuple[int, int]:
        """Map lon/lat to integer (row, col); ~inverse of the affine transform."""
        col, row = (~self.transform) * (lon, lat)
        return int(row), int(col)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.nrows and 0 <= col < self.ncols


def _metric_cell_size(transform, lat_center: float) -> tuple[float, float]:
    """Approximate cell width/height in metres for a lon/lat raster."""
    dlon = abs(transform.a)
    dlat = abs(transform.e)
    width_m = dlon * 111_320.0 * math.cos(math.radians(lat_center))
    height_m = dlat * 110_540.0
    return width_m, height_m


def condition_dem(dem_tif: str | Path) -> HydroLayers:
    """Fill sinks, compute D8 flow dir/accumulation, slope, TWI, depression depth."""
    import rasterio
    from pysheds.grid import Grid

    dem_tif = str(dem_tif)
    with rasterio.open(dem_tif) as src:
        transform = src.transform
        nodata = src.nodata
        bounds = src.bounds
    lat_center = (bounds.bottom + bounds.top) / 2.0
    cell_w, cell_h = _metric_cell_size(transform, lat_center)
    cell_area = cell_w * cell_h

    grid = Grid.from_raster(dem_tif)
    dem = grid.read_raster(dem_tif)

    # Replace nodata with the minimum valid elevation so borders act as low ground.
    dem_arr = np.asarray(dem, dtype="float64")
    if nodata is not None:
        mask = dem_arr == nodata
        if mask.any():
            valid_min = np.nanmin(dem_arr[~mask]) if (~mask).any() else 0.0
            dem_arr[mask] = valid_min
            dem = dem.copy()
            dem[...] = dem_arr

    # --- pysheds conditioning pipeline ---
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)

    filled_arr = np.asarray(flooded, dtype="float64")
    depression = np.clip(filled_arr - dem_arr, 0.0, None)

    # Slope (radians) from the conditioned DEM via metric gradients.
    dz_dy, dz_dx = np.gradient(inflated.astype("float64"), cell_h, cell_w)
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    # TWI = ln( specific catchment area / tan(slope) ).
    acc_arr = np.asarray(acc, dtype="float64")
    sca = (acc_arr + 1.0) * cell_area / cell_w  # specific catchment area (m)
    tan_slope = np.tan(np.clip(slope, 0.001, None))
    twi = np.log(sca / tan_slope)

    return HydroLayers(
        dem=dem_arr,
        filled=filled_arr,
        flow_dir=np.asarray(fdir),
        flow_acc=acc_arr,
        slope=slope,
        twi=twi,
        depression_depth=depression,
        transform=transform,
        cell_area_m2=cell_area,
        cell_width_m=cell_w,
        nrows=dem_arr.shape[0],
        ncols=dem_arr.shape[1],
    )


def sample_layers_at(layers: HydroLayers, lon: float, lat: float) -> dict:
    """Sample all terrain features at a lon/lat point (nearest cell)."""
    row, col = layers.rowcol(lon, lat)
    if not layers.in_bounds(row, col):
        return {k: None for k in
                ("elevation", "slope", "twi", "depression_depth", "upstream_area")}
    return {
        "elevation": float(layers.dem[row, col]),
        "slope": float(layers.slope[row, col]),
        "twi": float(layers.twi[row, col]),
        "depression_depth": float(layers.depression_depth[row, col]),
        "upstream_area": float(layers.flow_acc[row, col] * layers.cell_area_m2),
    }
