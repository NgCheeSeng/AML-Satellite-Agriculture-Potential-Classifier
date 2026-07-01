"""Monthly Google Earth Engine extraction for Sagil grid cells."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.validation import (
    build_month_schedule,
    build_square_cell,
    display_path,
    load_coordinates,
    load_project_config,
    next_month_start,
    path_from_config,
    project_path,
    utc_timestamp,
    validate_cell_area,
    write_json,
)

S2_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
S1_DATASET = "COPERNICUS/S1_GRD"
CHIRPS_DATASET = "UCSB-CHG/CHIRPS/DAILY"
ERA5_DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
SRTM_DATASET = "USGS/SRTMGL1_003"
DYNAMICWORLD_DATASET = "GOOGLE/DYNAMICWORLD/V1"
JRC_WATER_DATASET = "JRC/GSW1_4/GlobalSurfaceWater"

IDENTIFIER_COLUMNS = ["sample_id", "grid_row", "grid_col", "latitude", "longitude", "month"]
MONTHLY_OBSERVATION_COLUMNS = IDENTIFIER_COLUMNS + [
    "ndvi_mean", "evi_mean", "ndmi_mean", "ndwi_mean", "s2_image_count", "s2_valid_pixel_fraction",
    "vv_mean_db", "vh_mean_db", "vv_minus_vh_db", "vv_vh_ratio_linear", "s1_image_count",
    "rainfall_total_mm", "rainfall_mean_daily_mm", "rainfall_max_1day_mm", "dry_days_count", "heavy_rain_days_count",
    "temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_total_m", "evaporation_total_m",
    "water_probability_mean", "flooded_vegetation_probability_mean", "built_probability_mean", "dynamicworld_image_count",
    "elevation_mean_m", "elevation_min_m", "elevation_max_m", "slope_mean_deg", "lowland_fraction",
    "max_water_extent_fraction",
    "optical_missing_flag", "sar_missing_flag", "climate_missing_flag", "any_missing_flag",
]


@dataclass(frozen=True)
class MonthlyExtractionConfig:
    """Resolved settings for monthly GEE extraction."""

    config_path: str = "config/project_config.yaml"
    force: bool = False
    sample_limit: int | None = None
    verbose: bool = True


def load_gee_project_id(config: dict[str, Any]) -> str | None:
    """Load the Earth Engine project id from env or ignored local credentials."""

    gee_config = config.get("gee", {})
    env_name = gee_config.get("project_id_env", "GEE_PROJECT_ID")
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value

    credentials_path = gee_config.get("credentials_path")
    if credentials_path:
        resolved = project_path(credentials_path)
        if resolved.exists():
            import json

            data = json.loads(resolved.read_text(encoding="utf-8"))
            project_id = str(data.get("project_id") or data.get("gee_project_id") or "").strip()
            if project_id:
                return project_id
    return None


def initialize_earth_engine(config: dict[str, Any], ee_module: Any | None = None) -> Any:
    """Authenticate and initialize Earth Engine."""

    ee = ee_module
    if ee is None:
        import ee as ee_module_import

        ee = ee_module_import
    project_id = load_gee_project_id(config)
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    return ee


def extract_monthly_observations(
    extraction_config: MonthlyExtractionConfig | None = None,
    ee_module: Any | None = None,
) -> pd.DataFrame:
    """Extract monthly GEE observations and write raw per-cell CSVs."""

    extraction_config = extraction_config or MonthlyExtractionConfig()
    config = load_project_config(extraction_config.config_path)
    coordinates = load_coordinates(path_from_config(config, "paths", "coordinates_csv"))
    if extraction_config.sample_limit is not None:
        coordinates = coordinates.head(extraction_config.sample_limit)

    gee_config = config.get("gee", {})
    grid_config = config.get("grid", {})
    thresholds = config.get("thresholds", {})
    months = build_month_schedule(gee_config["first_month"], gee_config["last_month"])
    ee = initialize_earth_engine(config, ee_module=ee_module)

    raw_dir = path_from_config(config, "paths", "raw_monthly_dir")
    sample_index_csv = path_from_config(config, "paths", "sample_index_csv")
    manifest_json = path_from_config(config, "paths", "extraction_manifest_json")
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_index_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    iterator = _progress(coordinates.iterrows(), len(coordinates), "GEE cells", extraction_config.verbose)
    for _, coordinate in iterator:
        sample_id = str(coordinate["sample_id"])
        raw_csv = raw_dir / f"{sample_id}.csv"
        cell = build_square_cell(
            float(coordinate["latitude"]),
            float(coordinate["longitude"]),
            float(grid_config.get("half_width_m", 500)),
            str(grid_config.get("projected_crs", "EPSG:32648")),
        )
        validate_cell_area(
            float(cell["cell_area_m2"]),
            float(grid_config.get("expected_area_m2", 1_000_000)),
            float(grid_config.get("area_tolerance_fraction", 0.05)),
        )

        status = "cached"
        row_count = 0
        if raw_csv.exists() and not extraction_config.force:
            try:
                row_count = len(pd.read_csv(raw_csv))
            except Exception:
                row_count = 0
        else:
            try:
                data = extract_cell_monthly_rows(ee, coordinate, cell["wgs84_ring"], months, config)
                data.to_csv(raw_csv, index=False)
                row_count = len(data)
                status = "ok"
            except Exception as exc:
                status = "failed"
                failures.append({"sample_id": sample_id, "error": str(exc)})

        rows.append({
            "sample_id": sample_id,
            "grid_row": int(coordinate["grid_row"]),
            "grid_col": int(coordinate["grid_col"]),
            "latitude": float(coordinate["latitude"]),
            "longitude": float(coordinate["longitude"]),
            "cell_area_m2": float(cell["cell_area_m2"]),
            "first_month": months[0],
            "last_month": months[-1],
            "row_count": int(row_count),
            "raw_csv": display_path(raw_csv),
            "extraction_status": status,
        })

    sample_index = pd.DataFrame(rows)
    sample_index.to_csv(sample_index_csv, index=False)
    write_json(manifest_json, {
        "extraction_timestamp": utc_timestamp(),
        "study_area": config.get("project", {}).get("study_area"),
        "gee_project_id": load_gee_project_id(config),
        "first_month": months[0],
        "last_month": months[-1],
        "month_count": len(months),
        "monthly_interval_rule": "[month_start, next_month_start)",
        "raw_monthly_dir": display_path(raw_dir),
        "sample_count": int(len(sample_index)),
        "failed_samples": failures,
        "datasets": {
            "sentinel_2": S2_DATASET,
            "sentinel_1": S1_DATASET,
            "chirps": CHIRPS_DATASET,
            "era5_land": ERA5_DATASET,
            "srtm": SRTM_DATASET,
            "dynamic_world": DYNAMICWORLD_DATASET,
            "jrc_water": JRC_WATER_DATASET,
        },
        "thresholds": thresholds,
    })
    return sample_index


def extract_cell_monthly_rows(
    ee: Any,
    coordinate: pd.Series,
    wgs84_ring: list[list[float]],
    months: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Extract every monthly row for one grid cell."""

    region = ee.Geometry.Polygon([wgs84_ring])
    static_values = {}
    static_values.update(_extract_terrain(ee, region, config))
    static_values.update(_extract_historical_water(ee, region))

    rows = []
    for month in months:
        month_end = next_month_start(month)
        row: dict[str, Any] = {
            "sample_id": coordinate["sample_id"],
            "grid_row": int(coordinate["grid_row"]),
            "grid_col": int(coordinate["grid_col"]),
            "latitude": float(coordinate["latitude"]),
            "longitude": float(coordinate["longitude"]),
            "month": month,
        }
        row.update(_safe_group(lambda: _extract_s2(ee, region, month, month_end, config), _s2_nan()))
        row.update(_safe_group(lambda: _extract_s1(ee, region, month, month_end), _s1_nan()))
        row.update(_safe_group(lambda: _extract_chirps(ee, region, month, month_end, config), _chirps_nan()))
        row.update(_safe_group(lambda: _extract_era5(ee, region, month, month_end), _era5_nan()))
        row.update(_safe_group(lambda: _extract_dynamic_world(ee, region, month, month_end), _dynamic_world_nan()))
        row.update(static_values)
        row.update(_missing_flags(row))
        rows.append({column: row.get(column, np.nan) for column in MONTHLY_OBSERVATION_COLUMNS})
    return pd.DataFrame(rows, columns=MONTHLY_OBSERVATION_COLUMNS)


def _extract_s2(ee: Any, region: Any, start: str, end: str, config: dict[str, Any]) -> dict[str, Any]:
    gee_config = config.get("gee", {})
    reducer_name = str(gee_config.get("s2_composite_reducer", "median")).lower()
    collection = (
        ee.ImageCollection(S2_DATASET)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", float(gee_config.get("s2_cloud_probability_max", 80))))
        .map(_mask_s2_clouds)
        .map(lambda image: _add_s2_indices(ee, image))
    )
    count = _safe_size(collection)
    if count == 0:
        return _s2_nan() | {"s2_image_count": 0}
    composite = collection.median() if reducer_name == "median" else collection.mean()
    stats = _reduce_mean(ee, composite, region, 20, ["ndvi", "evi", "ndmi", "ndwi"])
    valid_fraction = _reduce_mean(
        ee,
        composite.select("ndvi").mask().rename("s2_valid_pixel_fraction"),
        region,
        20,
        ["s2_valid_pixel_fraction"],
    )
    return {
        "ndvi_mean": stats.get("ndvi"),
        "evi_mean": stats.get("evi"),
        "ndmi_mean": stats.get("ndmi"),
        "ndwi_mean": stats.get("ndwi"),
        "s2_image_count": count,
        "s2_valid_pixel_fraction": valid_fraction.get("s2_valid_pixel_fraction"),
    }


def _mask_s2_clouds(image: Any) -> Any:
    scl = image.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(mask)


def _add_s2_indices(ee: Any, image: Any) -> Any:
    scaled = image.select(["B2", "B3", "B4", "B8", "B11"]).multiply(0.0001)
    blue = scaled.select("B2")
    green = scaled.select("B3")
    red = scaled.select("B4")
    nir = scaled.select("B8")
    swir = scaled.select("B11")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": nir, "RED": red, "BLUE": blue},
    ).rename("evi")
    ndmi = nir.subtract(swir).divide(nir.add(swir)).rename("ndmi")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("ndwi")
    return image.addBands(ee.Image.cat([ndvi, evi, ndmi, ndwi]))


def _extract_s1(ee: Any, region: Any, start: str, end: str) -> dict[str, Any]:
    collection = (
        ee.ImageCollection(S1_DATASET)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    count = _safe_size(collection)
    if count == 0:
        return _s1_nan() | {"s1_image_count": 0}
    mean_image = collection.mean()
    image = ee.Image.cat([
        mean_image.select("VV").rename("vv_mean_db"),
        mean_image.select("VH").rename("vh_mean_db"),
        mean_image.select("VV").subtract(mean_image.select("VH")).rename("vv_minus_vh_db"),
    ])
    stats = _reduce_mean(ee, image, region, 30, ["vv_mean_db", "vh_mean_db", "vv_minus_vh_db"])
    vv_minus_vh = stats.get("vv_minus_vh_db")
    ratio = None if vv_minus_vh is None else float(np.power(10.0, vv_minus_vh / 10.0))
    stats.update({"vv_vh_ratio_linear": ratio, "s1_image_count": count})
    return stats


def _extract_chirps(ee: Any, region: Any, start: str, end: str, config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config.get("thresholds", {})
    dry_threshold = float(thresholds.get("dry_day_mm", 1.0))
    heavy_threshold = float(thresholds.get("heavy_rain_day_mm", 20.0))
    collection = ee.ImageCollection(CHIRPS_DATASET).filterBounds(region).filterDate(start, end).select("precipitation")
    count = _safe_size(collection)
    if count == 0:
        return _chirps_nan()
    image = ee.Image.cat([
        collection.sum().rename("rainfall_total_mm"),
        collection.mean().rename("rainfall_mean_daily_mm"),
        collection.max().rename("rainfall_max_1day_mm"),
        collection.map(lambda image: image.lt(dry_threshold).rename("dry_days_count")).sum(),
        collection.map(lambda image: image.gt(heavy_threshold).rename("heavy_rain_days_count")).sum(),
    ])
    return _reduce_mean(ee, image, region, 5500, [
        "rainfall_total_mm", "rainfall_mean_daily_mm", "rainfall_max_1day_mm", "dry_days_count", "heavy_rain_days_count",
    ])


def _extract_era5(ee: Any, region: Any, start: str, end: str) -> dict[str, Any]:
    collection = ee.ImageCollection(ERA5_DATASET).filterBounds(region).filterDate(start, end)
    if _safe_size(collection) == 0:
        return _era5_nan()
    mean_image = collection.mean()
    temp_c = mean_image.select("temperature_2m").subtract(273.15).rename("temperature_2m_mean_c")
    dew_c = mean_image.select("dewpoint_temperature_2m").subtract(273.15)
    rh = ee.Image().expression(
        "100 * (exp((17.625 * D) / (243.04 + D)) / exp((17.625 * T) / (243.04 + T)))",
        {"D": dew_c, "T": temp_c},
    ).rename("relative_humidity_mean_pct")
    soil = mean_image.select("volumetric_soil_water_layer_1").rename("soil_water_layer1_mean")
    runoff = collection.select("surface_runoff_sum").sum().rename("surface_runoff_total_m")
    evaporation = collection.select("total_evaporation_sum").sum().rename("evaporation_total_m")
    image = ee.Image.cat([temp_c, rh, soil, runoff, evaporation])
    return _reduce_mean(ee, image, region, 9000, [
        "temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_total_m", "evaporation_total_m",
    ])


def _extract_dynamic_world(ee: Any, region: Any, start: str, end: str) -> dict[str, Any]:
    collection = (
        ee.ImageCollection(DYNAMICWORLD_DATASET)
        .filterBounds(region)
        .filterDate(start, end)
        .select(["water", "flooded_vegetation", "built"])
    )
    count = _safe_size(collection)
    if count == 0:
        return _dynamic_world_nan() | {"dynamicworld_image_count": 0}
    image = collection.mean().rename([
        "water_probability_mean",
        "flooded_vegetation_probability_mean",
        "built_probability_mean",
    ])
    stats = _reduce_mean(ee, image, region, 10, [
        "water_probability_mean", "flooded_vegetation_probability_mean", "built_probability_mean",
    ])
    stats["dynamicworld_image_count"] = count
    return stats


def _extract_terrain(ee: Any, region: Any, config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config.get("thresholds", {})
    lowland_elevation = float(thresholds.get("lowland_elevation_m", 50.0))
    elevation = ee.Image(SRTM_DATASET).select("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope_mean_deg")
    lowland = elevation.lt(lowland_elevation).rename("lowland_fraction")
    mean_stats = _reduce_mean(ee, ee.Image.cat([
        elevation.rename("elevation_mean_m"),
        slope,
        lowland,
    ]), region, 30, ["elevation_mean_m", "slope_mean_deg", "lowland_fraction"])
    minmax_stats = _reduce_min_max(ee, elevation.rename("elevation"), region, 30, "elevation")
    return {
        "elevation_mean_m": mean_stats.get("elevation_mean_m"),
        "elevation_min_m": minmax_stats.get("elevation_min_m"),
        "elevation_max_m": minmax_stats.get("elevation_max_m"),
        "slope_mean_deg": mean_stats.get("slope_mean_deg"),
        "lowland_fraction": mean_stats.get("lowland_fraction"),
    }


def _extract_historical_water(ee: Any, region: Any) -> dict[str, Any]:
    stats = _reduce_mean(ee, ee.Image(JRC_WATER_DATASET).select("max_extent"), region, 30, ["max_extent"])
    return {"max_water_extent_fraction": stats.get("max_extent")}


def _safe_group(extractor: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return extractor()
    except Exception:
        return fallback.copy()


def _missing_flags(row: dict[str, Any]) -> dict[str, int]:
    optical_missing = int(pd.isna(row.get("ndvi_mean")) or float(row.get("s2_image_count") or 0) == 0)
    sar_missing = int(pd.isna(row.get("vv_mean_db")) or float(row.get("s1_image_count") or 0) == 0)
    climate_missing = int(pd.isna(row.get("rainfall_total_mm")) or pd.isna(row.get("temperature_2m_mean_c")))
    return {
        "optical_missing_flag": optical_missing,
        "sar_missing_flag": sar_missing,
        "climate_missing_flag": climate_missing,
        "any_missing_flag": int(any([optical_missing, sar_missing, climate_missing])),
    }


def _safe_size(collection: Any) -> int:
    try:
        return int(collection.size().getInfo())
    except Exception:
        return 0


def _reduce_mean(ee: Any, image: Any, region: Any, scale: int, keys: list[str]) -> dict[str, float | None]:
    try:
        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
        ).getInfo()
    except Exception:
        values = {}
    return {key: _as_float(values.get(key)) for key in keys}


def _reduce_min_max(ee: Any, image: Any, region: Any, scale: int, prefix: str) -> dict[str, float | None]:
    try:
        values = image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
        ).getInfo()
    except Exception:
        values = {}
    return {
        f"{prefix}_min_m": _as_float(values.get(f"{prefix}_min")),
        f"{prefix}_max_m": _as_float(values.get(f"{prefix}_max")),
    }


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _nan(keys: list[str]) -> dict[str, float]:
    return {key: np.nan for key in keys}


def _s2_nan() -> dict[str, Any]:
    return _nan(["ndvi_mean", "evi_mean", "ndmi_mean", "ndwi_mean", "s2_valid_pixel_fraction"]) | {"s2_image_count": 0}


def _s1_nan() -> dict[str, Any]:
    return _nan(["vv_mean_db", "vh_mean_db", "vv_minus_vh_db", "vv_vh_ratio_linear"]) | {"s1_image_count": 0}


def _chirps_nan() -> dict[str, Any]:
    return _nan(["rainfall_total_mm", "rainfall_mean_daily_mm", "rainfall_max_1day_mm", "dry_days_count", "heavy_rain_days_count"])


def _era5_nan() -> dict[str, Any]:
    return _nan(["temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_total_m", "evaporation_total_m"])


def _dynamic_world_nan() -> dict[str, Any]:
    return _nan(["water_probability_mean", "flooded_vegetation_probability_mean", "built_probability_mean"]) | {"dynamicworld_image_count": 0}


def _progress(iterable: Any, total: int, desc: str, verbose: bool) -> Any:
    if not verbose:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)
