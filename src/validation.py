"""Validation and configuration helpers for the Sagil monthly GEE dataset."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REQUIRED_COORDINATE_COLUMNS = ["grid_row", "grid_col", "latitude", "longitude"]
DEFAULT_CONFIG_PATH = Path("config") / "project_config.yaml"


def project_root() -> Path:
    """Return the current project root, including when called from notebooks/."""

    cwd = Path.cwd().resolve()
    return cwd.parent if cwd.name == "notebooks" else cwd


def project_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a project-relative path."""

    value = Path(path)
    if value.is_absolute():
        return value
    base = Path(root).resolve() if root is not None else project_root()
    return base / value


def display_path(path: str | Path, root: str | Path | None = None) -> str:
    """Format a path relative to the project root when possible."""

    base = Path(root).resolve() if root is not None else project_root()
    resolved = project_path(path, base).resolve()
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return str(path)


def load_project_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML project configuration."""

    resolved = project_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing project config: {display_path(resolved)}")
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def path_from_config(config: dict[str, Any], section: str, key: str) -> Path:
    """Resolve one path from the config file."""

    try:
        return project_path(config[section][key])
    except KeyError as exc:
        raise KeyError(f"Missing config path: {section}.{key}") from exc


def stable_sample_id(row: pd.Series) -> str:
    """Return a deterministic sample id for one coordinate row."""

    raw_sample_id = str(row.get("sample_id", "")).strip()
    if raw_sample_id and raw_sample_id.lower() != "nan":
        return sanitize_sample_id(raw_sample_id)
    if pd.notna(row.get("grid_row")) and pd.notna(row.get("grid_col")):
        return f"sagil_r{int(row['grid_row']):02d}_c{int(row['grid_col']):02d}"
    return sanitize_sample_id(f"{float(row['latitude']):.6f}_{float(row['longitude']):.6f}")


def sanitize_sample_id(value: str) -> str:
    """Make a sample id safe for stable CSV filenames."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_")


def load_coordinates(coordinates_csv: str | Path) -> pd.DataFrame:
    """Load and validate the coordinate source CSV."""

    path = project_path(coordinates_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing coordinates file: {display_path(path)}")
    coordinates = pd.read_csv(path)
    if coordinates.empty:
        raise ValueError(f"Coordinate file is empty: {display_path(path)}")
    missing = [column for column in REQUIRED_COORDINATE_COLUMNS if column not in coordinates.columns]
    if missing:
        raise ValueError(f"Coordinate file is missing required columns: {missing}")

    coordinates = coordinates.copy()
    coordinates["sample_id"] = coordinates.apply(stable_sample_id, axis=1)
    coordinates["grid_row"] = coordinates["grid_row"].astype(int)
    coordinates["grid_col"] = coordinates["grid_col"].astype(int)
    coordinates["latitude"] = coordinates["latitude"].astype(float)
    coordinates["longitude"] = coordinates["longitude"].astype(float)

    if coordinates["sample_id"].duplicated().any():
        duplicates = coordinates.loc[coordinates["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values in coordinates.csv: {duplicates[:10]}")
    if coordinates[["grid_row", "grid_col"]].duplicated().any():
        raise ValueError("Duplicate grid_row/grid_col pairs in coordinates.csv")
    if coordinates[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Coordinate file contains missing latitude or longitude values")
    return coordinates


def build_month_schedule(first_month: str, last_month: str) -> list[str]:
    """Build inclusive calendar-month starts."""

    start = pd.Timestamp(first_month).normalize()
    end = pd.Timestamp(last_month).normalize()
    if start.day != 1 or end.day != 1:
        raise ValueError("FIRST_MONTH and LAST_MONTH must both be month starts: YYYY-MM-01")
    if end < start:
        raise ValueError("LAST_MONTH must be on or after FIRST_MONTH")
    return [value.date().isoformat() for value in pd.date_range(start=start, end=end, freq="MS")]


def next_month_start(month: str) -> str:
    """Return the exclusive end date for a month-start label."""

    return (pd.Timestamp(month) + pd.DateOffset(months=1)).date().isoformat()


def build_square_cell(latitude: float, longitude: float, half_width_m: float, projected_crs: str) -> dict[str, Any]:
    """Build a square cell around a centroid using a projected CRS."""

    from pyproj import Transformer

    to_projected = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)

    x_center, y_center = to_projected.transform(longitude, latitude)
    corners_projected = [
        (x_center - half_width_m, y_center - half_width_m),
        (x_center + half_width_m, y_center - half_width_m),
        (x_center + half_width_m, y_center + half_width_m),
        (x_center - half_width_m, y_center + half_width_m),
        (x_center - half_width_m, y_center - half_width_m),
    ]
    ring_wgs84 = []
    for x_value, y_value in corners_projected:
        lon_value, lat_value = to_wgs84.transform(x_value, y_value)
        ring_wgs84.append([lon_value, lat_value])
    side_length = 2 * float(half_width_m)
    return {
        "projected_ring": corners_projected,
        "wgs84_ring": ring_wgs84,
        "cell_area_m2": side_length * side_length,
    }


def validate_cell_area(cell_area_m2: float, expected_area_m2: float, tolerance_fraction: float) -> None:
    """Validate that a cell area is within the configured tolerance."""

    tolerance = expected_area_m2 * tolerance_fraction
    if math.fabs(cell_area_m2 - expected_area_m2) > tolerance:
        raise ValueError(
            f"Cell area {cell_area_m2:.2f} m2 is outside tolerance around {expected_area_m2:.2f} m2"
        )


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write pretty JSON to a project path."""

    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")
