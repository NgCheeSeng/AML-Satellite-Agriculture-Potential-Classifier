"""Feature extraction helpers for sustainable agriculture modeling.

Future/t+1 values are written only to gee_targets.csv. They are never written to
gee_features.csv, so model input files cannot silently include future data.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

S2_DATASET = "COPERNICUS/S2_SR_HARMONIZED"
S1_DATASET = "COPERNICUS/S1_GRD"
CHIRPS_DATASET = "UCSB-CHG/CHIRPS/DAILY"
ERA5_DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
SRTM_DATASET = "USGS/SRTMGL1_003"
DYNAMICWORLD_DATASET = "GOOGLE/DYNAMICWORLD/V1"
JRC_WATER_DATASET = "JRC/GSW1_4/GlobalSurfaceWater"

ID_COLUMNS = ["sample_id", "label", "latitude", "longitude", "frame_index", "acquisition_date"]
OBSERVATION_COLUMNS = ID_COLUMNS + [
    "ndvi_mean", "evi_mean", "ndwi_mean", "ndmi_mean", "s2_image_count",
    "rainfall_30d_mm", "rainfall_90d_mm", "dry_days_30d", "heavy_rain_days_30d", "heavy_rain_days_90d",
    "temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_30d_m",
    "elevation_m", "slope_deg", "lowland_flag",
    "vv_mean_db", "vh_mean_db", "vv_minus_vh_db", "s1_image_count",
    "built_probability_1km", "built_probability_5km", "flooded_vegetation_probability", "dynamicworld_image_count",
    "water_occurrence_mean", "max_water_extent_fraction",
]
FEATURE_COLUMNS = ID_COLUMNS + [
    "ndvi_lag_1", "ndvi_lag_2", "ndvi_rolling_mean_3", "ndvi_trend",
    "evi_lag_1", "evi_lag_2", "evi_rolling_mean_3", "evi_trend",
    "built_growth_rate", "built_growth_trend", "rainfall_rolling_mean_3", "heavy_rain_rolling_sum_3",
    "sar_moisture_trend", "flood_risk_proxy_score",
]
TARGET_COLUMNS = [
    "sample_id", "label", "frame_index", "acquisition_date", "target_date",
    "target_ndvi_delta_1", "target_evi_delta_1", "target_built_delta_1", "target_sustainability_proxy_score",
]


@dataclass(frozen=True)
class FeatureExtractionConfig:
    gee_project_id: str | None = None
    data_dir: str = "data"
    sample_index_csv: str = "data/processed/sample_index.csv"
    buffer_radius_m: int = 1000
    context_buffer_radius_m: int = 5000
    s2_lookback_days: int = 45
    s1_tolerance_days: int = 15
    dynamicworld_lookback_days: int = 180
    dry_day_threshold_mm: float = 1.0
    heavy_rain_threshold_mm: float = 20.0
    lowland_elevation_threshold_m: float = 50.0
    force: bool = False
    verbose: bool = True
    log_feature_groups: bool = True

    @property
    def project_id(self) -> str | None:
        return self.gee_project_id or os.environ.get("GEE_PROJECT_ID") or None


def initialize_earth_engine(project_id: str | None = None):
    import ee

    resolved_project = project_id or os.environ.get("GEE_PROJECT_ID") or None
    try:
        if resolved_project:
            ee.Initialize(project=resolved_project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if resolved_project:
            ee.Initialize(project=resolved_project)
        else:
            ee.Initialize()
    return ee


def load_sample_index(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing sample index: {path}")
    return pd.read_csv(path)


def load_frame_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing frame metadata: {path}")
    return pd.read_csv(path)


def _log(config: FeatureExtractionConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _progress(iterable: Any, *, total: int | None, desc: str, config: FeatureExtractionConfig, leave: bool = True) -> Any:
    if not config.verbose:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave)


def _timed_group(config: FeatureExtractionConfig, sample_id: str, acquisition_date: str, group_name: str, extractor: Any) -> dict[str, Any]:
    if config.log_feature_groups:
        _log(config, f"      {sample_id} {acquisition_date}: {group_name} start")
    started_at = time.perf_counter()
    result = extractor()
    if config.log_feature_groups:
        elapsed = time.perf_counter() - started_at
        _log(config, f"      {sample_id} {acquisition_date}: {group_name} done ({elapsed:.1f}s)")
    return result


def extract_all_samples(config: FeatureExtractionConfig, sample_limit: int | None = None, ee_module: Any | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ee = ee_module or initialize_earth_engine(config.project_id)
    sample_index = load_sample_index(config.sample_index_csv)
    if sample_limit is not None:
        sample_index = sample_index.head(sample_limit)

    total_samples = len(sample_index)
    _log(config, f"Starting GEE extraction: {total_samples} sample(s), project={config.project_id}")
    _log(config, f"Outputs will be written under: {Path(config.data_dir) / 'processed'}")

    feature_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    sample_iter = _progress(sample_index.iterrows(), total=total_samples, desc="GEE samples", config=config)
    for sample_number, (_, sample) in enumerate(sample_iter, start=1):
        sample_id = str(sample["sample_id"])
        sample_started_at = time.perf_counter()
        _log(config, f"[{sample_number}/{total_samples}] {sample_id}: start")
        observations = extract_sample_observations(sample, config, ee)
        _log(config, f"[{sample_number}/{total_samples}] {sample_id}: building temporal features/targets")
        features, targets = build_features_and_targets(observations)
        write_sample_outputs(sample, observations, features, targets, config)
        feature_frames.append(features)
        target_frames.append(targets)
        elapsed = time.perf_counter() - sample_started_at
        _log(config, f"[{sample_number}/{total_samples}] {sample_id}: done ({len(observations)} frame rows, {elapsed:.1f}s)")

    features_all = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(columns=FEATURE_COLUMNS)
    targets_all = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame(columns=TARGET_COLUMNS)
    assert_no_target_leakage(features_all)

    processed_root = Path(config.data_dir) / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    features_path = processed_root / "gee_features_all.csv"
    targets_path = processed_root / "gee_targets_all.csv"
    features_all.to_csv(features_path, index=False)
    targets_all.to_csv(targets_path, index=False)
    _log(config, f"Saved combined features: {features_path} ({len(features_all)} rows)")
    _log(config, f"Saved combined targets: {targets_path} ({len(targets_all)} rows)")
    return features_all, targets_all

def extract_sample_observations(sample: pd.Series, config: FeatureExtractionConfig, ee: Any) -> pd.DataFrame:
    output_path = Path(str(sample["gee_observations_csv"]))
    sample_id = str(sample["sample_id"])
    if output_path.exists() and not config.force:
        _log(config, f"    {sample_id}: using cached observations from {output_path}")
        return pd.read_csv(output_path)

    frame_metadata = load_frame_metadata(sample["frame_metadata_csv"])
    total_frames = len(frame_metadata)
    _log(config, f"    {sample_id}: extracting {total_frames} frame date(s)")
    rows = []
    frame_iter = _progress(frame_metadata.iterrows(), total=total_frames, desc=f"{sample_id} frames", config=config, leave=False)
    for frame_number, (_, frame) in enumerate(frame_iter, start=1):
        acquisition_date = str(frame["acquisition_date"])
        frame_started_at = time.perf_counter()
        _log(config, f"    {sample_id}: frame {frame_number}/{total_frames} {acquisition_date} start")
        rows.append(extract_observation(sample, frame, acquisition_date, config, ee))
        elapsed = time.perf_counter() - frame_started_at
        _log(config, f"    {sample_id}: frame {frame_number}/{total_frames} {acquisition_date} done ({elapsed:.1f}s)")
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)


def extract_observation(sample: pd.Series, frame: pd.Series, acquisition_date: str, config: FeatureExtractionConfig, ee: Any) -> dict[str, Any]:
    latitude = float(sample["latitude"])
    longitude = float(sample["longitude"])
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(config.buffer_radius_m)
    context_region = point.buffer(config.context_buffer_radius_m)
    current_date = date.fromisoformat(acquisition_date)

    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "latitude": latitude,
        "longitude": longitude,
        "frame_index": int(frame["frame_index"]),
        "acquisition_date": acquisition_date,
    }
    sample_id = str(sample["sample_id"])
    row.update(_timed_group(config, sample_id, acquisition_date, "Sentinel-2 vegetation", lambda: _extract_s2(ee, region, current_date, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "CHIRPS rainfall", lambda: _extract_chirps(ee, region, current_date, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "ERA5-Land climate", lambda: _extract_era5(ee, region, current_date)))
    row.update(_timed_group(config, sample_id, acquisition_date, "SRTM terrain", lambda: _extract_terrain(ee, region, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "Sentinel-1 SAR", lambda: _extract_s1(ee, region, current_date, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "Dynamic World urban/water", lambda: _extract_dynamic_world(ee, region, context_region, current_date, config)))
    row.update(_timed_group(config, sample_id, acquisition_date, "JRC water", lambda: _extract_water(ee, region)))
    return {column: row.get(column, np.nan) for column in OBSERVATION_COLUMNS}


def build_features_and_targets(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if observations.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.DataFrame(columns=TARGET_COLUMNS)

    df = observations.sort_values(["sample_id", "acquisition_date"]).copy()
    group = df.groupby("sample_id", group_keys=False)
    features = df[ID_COLUMNS].copy()

    for prefix in ["ndvi", "evi"]:
        source = f"{prefix}_mean"
        features[f"{prefix}_lag_1"] = group[source].shift(1)
        features[f"{prefix}_lag_2"] = group[source].shift(2)
        features[f"{prefix}_rolling_mean_3"] = group[source].transform(lambda s: s.rolling(3, min_periods=1).mean())
        features[f"{prefix}_trend"] = group[source].transform(lambda s: s.diff(2) / 2.0)

    features["built_growth_rate"] = group["built_probability_5km"].diff()
    features["built_growth_trend"] = group["built_probability_5km"].transform(lambda s: s.diff().rolling(3, min_periods=1).mean())
    features["rainfall_rolling_mean_3"] = group["rainfall_30d_mm"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    features["heavy_rain_rolling_sum_3"] = group["heavy_rain_days_30d"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    features["sar_moisture_trend"] = group["vh_mean_db"].transform(lambda s: s.diff(2) / 2.0)
    features["flood_risk_proxy_score"] = _flood_risk_proxy(df)
    features = features.reindex(columns=FEATURE_COLUMNS)
    assert_no_target_leakage(features)

    targets = df[["sample_id", "label", "frame_index", "acquisition_date"]].copy()
    targets["target_date"] = group["acquisition_date"].shift(-1)
    targets["target_ndvi_delta_1"] = group["ndvi_mean"].shift(-1) - df["ndvi_mean"]
    targets["target_evi_delta_1"] = group["evi_mean"].shift(-1) - df["evi_mean"]
    targets["target_built_delta_1"] = group["built_probability_5km"].shift(-1) - df["built_probability_5km"]
    targets["target_sustainability_proxy_score"] = (
        targets["target_ndvi_delta_1"].fillna(0)
        + targets["target_evi_delta_1"].fillna(0)
        - targets["target_built_delta_1"].fillna(0)
        - features["flood_risk_proxy_score"].fillna(0) * 0.1
    )
    return features, targets.reindex(columns=TARGET_COLUMNS)


def write_sample_outputs(sample: pd.Series, observations: pd.DataFrame, features: pd.DataFrame, targets: pd.DataFrame, config: FeatureExtractionConfig) -> None:
    observations_path = Path(str(sample["gee_observations_csv"]))
    features_path = Path(str(sample["gee_features_csv"]))
    targets_path = Path(str(sample["gee_targets_csv"]))
    metadata_path = Path(str(sample["gee_feature_metadata_json"]))
    observations_path.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(observations_path, index=False)
    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path, index=False)
    metadata_path.write_text(json.dumps(build_feature_metadata(sample, observations, config), indent=2), encoding="utf-8")


def build_feature_metadata(sample: pd.Series, observations: pd.DataFrame, config: FeatureExtractionConfig) -> dict[str, Any]:
    dates = pd.to_datetime(observations["acquisition_date"], errors="coerce").dropna()
    end_date = dates.max().date().isoformat() if len(dates) else ""
    s2_start = (dates.min().date() - timedelta(days=config.s2_lookback_days)).isoformat() if len(dates) else ""
    chirps_start = (dates.min().date() - timedelta(days=90)).isoformat() if len(dates) else ""
    era5_start = (dates.min().date() - timedelta(days=30)).isoformat() if len(dates) else ""
    return {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "gee_project_id": config.project_id,
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "s2_date_range": {"start": s2_start, "end": end_date},
        "s1_tolerance_days": config.s1_tolerance_days,
        "chirps_date_range": {"start": chirps_start, "end": end_date},
        "era5_date_range": {"start": era5_start, "end": end_date},
        "srtm_version": SRTM_DATASET,
        "dynamicworld_version": DYNAMICWORLD_DATASET,
        "buffer_radius_m": config.buffer_radius_m,
        "context_buffer_radius_m": config.context_buffer_radius_m,
    }


def assert_no_target_leakage(features: pd.DataFrame) -> None:
    leaks = [c for c in features.columns if c.startswith("target_") or c.startswith("future_") or "delta_1" in c]
    if leaks:
        raise ValueError(f"Future target columns found in gee_features.csv: {leaks}")

def _extract_s2(ee: Any, region: Any, current_date: date, config: FeatureExtractionConfig) -> dict[str, Any]:
    collection = (
        ee.ImageCollection(S2_DATASET)
        .filterBounds(region)
        .filterDate(*_window(current_date, config.s2_lookback_days, 1))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(_mask_s2_clouds)
        .map(lambda image: _add_s2_indices(ee, image))
    )
    count = _safe_size(collection)
    if count == 0:
        return _nan_values(["ndvi_mean", "evi_mean", "ndwi_mean", "ndmi_mean"], {"s2_image_count": 0})
    stats = _reduce_mean(ee, collection.mean(), region, 20, ["ndvi", "evi", "ndwi", "ndmi"])
    return {"ndvi_mean": stats.get("ndvi"), "evi_mean": stats.get("evi"), "ndwi_mean": stats.get("ndwi"), "ndmi_mean": stats.get("ndmi"), "s2_image_count": count}


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
    evi = image.expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {"NIR": nir, "RED": red, "BLUE": blue}).rename("evi")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("ndwi")
    ndmi = nir.subtract(swir).divide(nir.add(swir)).rename("ndmi")
    return ee.Image.cat([ndvi, evi, ndwi, ndmi])


def _extract_chirps(ee: Any, region: Any, current_date: date, config: FeatureExtractionConfig) -> dict[str, Any]:
    collection = ee.ImageCollection(CHIRPS_DATASET).filterBounds(region)
    precip_30 = collection.filterDate(*_window(current_date, 30, 1)).select("precipitation")
    precip_90 = collection.filterDate(*_window(current_date, 90, 1)).select("precipitation")
    if _safe_size(precip_30) == 0:
        return _nan_values(["rainfall_30d_mm", "rainfall_90d_mm", "dry_days_30d", "heavy_rain_days_30d", "heavy_rain_days_90d"])
    dry_days = precip_30.map(lambda image: image.lt(config.dry_day_threshold_mm).rename("dry_days_30d")).sum()
    heavy_30 = precip_30.map(lambda image: image.gt(config.heavy_rain_threshold_mm).rename("heavy_rain_days_30d")).sum()
    heavy_90 = precip_90.map(lambda image: image.gt(config.heavy_rain_threshold_mm).rename("heavy_rain_days_90d")).sum()
    image = ee.Image.cat([
        precip_30.sum().rename("rainfall_30d_mm"),
        precip_90.sum().rename("rainfall_90d_mm"),
        dry_days,
        heavy_30,
        heavy_90,
    ])
    return _reduce_mean(ee, image, region, 5500, ["rainfall_30d_mm", "rainfall_90d_mm", "dry_days_30d", "heavy_rain_days_30d", "heavy_rain_days_90d"])


def _extract_era5(ee: Any, region: Any, current_date: date) -> dict[str, Any]:
    keys = ["temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_30d_m"]
    collection = ee.ImageCollection(ERA5_DATASET).filterBounds(region).filterDate(*_window(current_date, 30, 1))
    if _safe_size(collection) == 0:
        return _nan_values(keys)
    mean = collection.mean()
    temp_c = mean.select("temperature_2m").subtract(273.15).rename("temperature_2m_mean_c")
    dew_c = mean.select("dewpoint_temperature_2m").subtract(273.15)
    rh = ee.Image().expression("100 * (exp((17.625 * D) / (243.04 + D)) / exp((17.625 * T) / (243.04 + T)))", {"D": dew_c, "T": temp_c}).rename("relative_humidity_mean_pct")
    soil = mean.select("volumetric_soil_water_layer_1").rename("soil_water_layer1_mean")
    runoff = collection.select("surface_runoff_sum").sum().rename("surface_runoff_30d_m")
    return _reduce_mean(ee, ee.Image.cat([temp_c, rh, soil, runoff]), region, 9000, keys)


def _extract_terrain(ee: Any, region: Any, config: FeatureExtractionConfig) -> dict[str, Any]:
    elevation = ee.Image(SRTM_DATASET).select("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope_deg")
    stats = _reduce_mean(ee, ee.Image.cat([elevation.rename("elevation_m"), slope]), region, 30, ["elevation_m", "slope_deg"])
    elevation_value = stats.get("elevation_m")
    stats["lowland_flag"] = int(elevation_value is not None and elevation_value < config.lowland_elevation_threshold_m)
    return stats

def _extract_s1(ee: Any, region: Any, current_date: date, config: FeatureExtractionConfig) -> dict[str, Any]:
    collection = (
        ee.ImageCollection(S1_DATASET)
        .filterBounds(region)
        .filterDate(*_window(current_date, config.s1_tolerance_days, config.s1_tolerance_days + 1))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    count = _safe_size(collection)
    if count == 0:
        return _nan_values(["vv_mean_db", "vh_mean_db", "vv_minus_vh_db"], {"s1_image_count": 0})
    mean = collection.mean()
    image = ee.Image.cat([
        mean.select("VV").rename("vv_mean_db"),
        mean.select("VH").rename("vh_mean_db"),
        mean.select("VV").subtract(mean.select("VH")).rename("vv_minus_vh_db"),
    ])
    stats = _reduce_mean(ee, image, region, 30, ["vv_mean_db", "vh_mean_db", "vv_minus_vh_db"])
    stats["s1_image_count"] = count
    return stats


def _extract_dynamic_world(ee: Any, region: Any, context_region: Any, current_date: date, config: FeatureExtractionConfig) -> dict[str, Any]:
    collection = (
        ee.ImageCollection(DYNAMICWORLD_DATASET)
        .filterBounds(context_region)
        .filterDate(*_window(current_date, config.dynamicworld_lookback_days, 1))
        .select(["built", "flooded_vegetation"])
    )
    count = _safe_size(collection)
    if count == 0:
        return _nan_values(["built_probability_1km", "built_probability_5km", "flooded_vegetation_probability"], {"dynamicworld_image_count": 0})
    image = collection.mean()
    built_1km = _reduce_mean(ee, image.select("built"), region, 10, ["built"]).get("built")
    built_5km = _reduce_mean(ee, image.select("built"), context_region, 10, ["built"]).get("built")
    flooded = _reduce_mean(ee, image.select("flooded_vegetation"), region, 10, ["flooded_vegetation"]).get("flooded_vegetation")
    return {
        "built_probability_1km": built_1km,
        "built_probability_5km": built_5km,
        "flooded_vegetation_probability": flooded,
        "dynamicworld_image_count": count,
    }


def _extract_water(ee: Any, region: Any) -> dict[str, Any]:
    stats = _reduce_mean(ee, ee.Image(JRC_WATER_DATASET).select(["occurrence", "max_extent"]), region, 30, ["occurrence", "max_extent"])
    return {"water_occurrence_mean": stats.get("occurrence"), "max_water_extent_fraction": stats.get("max_extent")}


def _flood_risk_proxy(df: pd.DataFrame) -> pd.Series:
    heavy = _minmax(df.get("heavy_rain_days_90d"), df.index)
    water = _minmax(df.get("water_occurrence_mean"), df.index)
    flooded = _minmax(df.get("flooded_vegetation_probability"), df.index)
    lowland = df.get("lowland_flag", pd.Series(0, index=df.index)).fillna(0)
    return (0.35 * heavy + 0.25 * water + 0.25 * flooded + 0.15 * lowland).clip(0, 1)


def _minmax(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(0.0, index=index)
    series = values.astype(float)
    min_value = series.min(skipna=True)
    max_value = series.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def _window(current_date: date, days_before: int, days_after: int) -> tuple[str, str]:
    return (current_date - timedelta(days=days_before)).isoformat(), (current_date + timedelta(days=days_after)).isoformat()


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
    result = {}
    for key in keys:
        value = values.get(key)
        try:
            result[key] = None if value is None else float(value)
        except (TypeError, ValueError):
            result[key] = None
    return result


def _nan_values(keys: list[str], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    values = {key: np.nan for key in keys}
    if overrides:
        values.update(overrides)
    return values
