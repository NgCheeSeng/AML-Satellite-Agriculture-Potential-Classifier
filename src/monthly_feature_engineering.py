"""Deterministic feature engineering for monthly Sagil GEE observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.gee_monthly_extraction import MONTHLY_OBSERVATION_COLUMNS
from src.validation import build_month_schedule, display_path, load_project_config, path_from_config, project_path

HYDROLOGY_FEATURE_VARIABLES = [
    "rainfall_total_mm",
    "rainfall_max_1day_mm",
    "dry_days_count",
    "heavy_rain_days_count",
    "soil_water_layer1_mean",
    "surface_runoff_total_m",
    "evaporation_total_m",
    "ndvi_mean",
    "ndmi_mean",
    "vv_mean_db",
    "vh_mean_db",
    "water_probability_mean",
    "flooded_vegetation_probability_mean",
]
ROLLING_SUM_VARIABLES = [
    "rainfall_total_mm",
    "surface_runoff_total_m",
    "evaporation_total_m",
    "dry_days_count",
    "heavy_rain_days_count",
]
LAG_MONTHS = [1, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
DIFFERENCE_MONTHS = [1, 3]


def build_processed_dataset(config_path: str | Path = "config/project_config.yaml") -> pd.DataFrame:
    """Load raw monthly CSVs, create deterministic features, and write processed_data.csv."""

    config = load_project_config(config_path)
    sample_index_path = path_from_config(config, "paths", "sample_index_csv")
    processed_path = path_from_config(config, "paths", "processed_data_csv")
    if not sample_index_path.exists():
        raise FileNotFoundError(f"Missing sample index: {display_path(sample_index_path)}")

    sample_index = pd.read_csv(sample_index_path)
    frames = []
    for _, sample in sample_index.iterrows():
        raw_path = project_path(sample["raw_csv"])
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw monthly CSV for {sample['sample_id']}: {display_path(raw_path)}")
        frames.append(pd.read_csv(raw_path))

    if not frames:
        raise ValueError("No raw monthly CSVs were loaded")

    data = pd.concat(frames, ignore_index=True)
    data = validate_and_sort_monthly_data(data, config)
    processed = add_deterministic_features(data)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)
    return processed


def validate_and_sort_monthly_data(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Validate continuity and sort by sample_id/month."""

    missing = [column for column in ["sample_id", "month"] if column not in data.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    for column in MONTHLY_OBSERVATION_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    expected_months = build_month_schedule(config["gee"]["first_month"], config["gee"]["last_month"])
    expected_set = set(expected_months)
    observed_columns = [column for column in MONTHLY_OBSERVATION_COLUMNS if column in data.columns]
    working = data[observed_columns].copy()
    working["month"] = pd.to_datetime(working["month"], errors="coerce")
    if working["month"].isna().any():
        raise ValueError("Raw data contains invalid month values")
    working["month"] = working["month"].dt.date.astype(str)

    duplicate_keys = working.duplicated(["sample_id", "month"])
    if duplicate_keys.any():
        duplicates = working.loc[duplicate_keys, ["sample_id", "month"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate sample_id/month rows found: {duplicates}")

    for sample_id, sample_data in working.groupby("sample_id"):
        observed = set(sample_data["month"])
        missing_months = sorted(expected_set - observed)
        extra_months = sorted(observed - expected_set)
        if missing_months or extra_months:
            raise ValueError(
                f"Monthly continuity failed for {sample_id}: missing={missing_months[:5]}, extra={extra_months[:5]}"
            )

    return working.sort_values(["sample_id", "month"]).reset_index(drop=True)


def add_deterministic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create a compact hydrology-focused feature set without learned transforms."""

    features = data.copy()
    month_dt = pd.to_datetime(features["month"])
    features["year"] = month_dt.dt.year.astype(int)
    features["month_number"] = month_dt.dt.month.astype(int)
    features["month_sin"] = np.sin(2 * np.pi * features["month_number"] / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * features["month_number"] / 12.0)

    for variable in [column for column in HYDROLOGY_FEATURE_VARIABLES if column in features.columns]:
        features[variable] = pd.to_numeric(features[variable], errors="coerce")

    group = features.groupby("sample_id", group_keys=False)
    engineered: dict[str, pd.Series] = {}
    for variable in [column for column in HYDROLOGY_FEATURE_VARIABLES if column in features.columns]:
        for lag in LAG_MONTHS:
            engineered[f"{variable}_lag_{lag}"] = group[variable].shift(lag)
        for window in ROLLING_WINDOWS:
            engineered[f"{variable}_rolling_mean_{window}"] = group[variable].transform(
                lambda series, window=window: series.rolling(window, min_periods=1).mean()
            )
            if variable in ROLLING_SUM_VARIABLES:
                engineered[f"{variable}_rolling_sum_{window}"] = group[variable].transform(
                    lambda series, window=window: series.rolling(window, min_periods=1).sum()
                )
        for diff_months in DIFFERENCE_MONTHS:
            engineered[f"{variable}_difference_{diff_months}m"] = group[variable].diff(diff_months)
    if engineered:
        features = pd.concat([features, pd.DataFrame(engineered, index=features.index)], axis=1)
    return features.copy()
