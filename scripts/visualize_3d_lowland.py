"""Show an interactive 3D lowland/elevation plot from raw monthly GEE CSV files.

The raw monthly CSVs repeat static terrain values for every month. This script
extracts one terrain row per sample_id and plots {latitude, longitude, height}
with color/marker size based on lowland_fraction. It does not write files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize


STATIC_COLUMNS = [
    "sample_id",
    "grid_row",
    "grid_col",
    "latitude",
    "longitude",
    "elevation_mean_m",
    "elevation_min_m",
    "elevation_max_m",
    "slope_mean_deg",
    "lowland_fraction",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Show a 3D lowland/elevation plot from raw monthly CSV files.")
    parser.add_argument("--raw-dir", default="data/raw/monthly", help="Folder containing raw monthly CSV files.")
    parser.add_argument(
        "--height-column",
        default="elevation_mean_m",
        choices=["elevation_mean_m", "elevation_min_m", "elevation_max_m"],
        help="Terrain column used as the z-axis height.",
    )
    parser.add_argument(
        "--lowland-threshold",
        type=float,
        default=0.70,
        help="Highlight cells at or above this lowland fraction with red marker edges.",
    )
    parser.add_argument("--azim", type=float, default=-55.0, help="3D plot azimuth view angle.")
    parser.add_argument("--elev", type=float, default=30.0, help="3D plot elevation view angle.")
    return parser.parse_args()


def first_valid_value(series: pd.Series):
    """Return the first non-null value from a repeated monthly column."""

    valid = series.dropna()
    return np.nan if valid.empty else valid.iloc[0]


def load_lowland_points(raw_dir: str | Path, height_column: str) -> pd.DataFrame:
    """Return one {latitude, longitude, height} row per grid-cell CSV."""

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw monthly folder not found: {raw_path}")

    rows = []
    for csv_path in sorted(raw_path.glob("*.csv")):
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        required = ["sample_id", "latitude", "longitude", height_column, "lowland_fraction"]
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        usecols = [column for column in STATIC_COLUMNS if column in header]
        data = pd.read_csv(csv_path, usecols=usecols)
        row = {column: first_valid_value(data[column]) for column in usecols}
        rows.append(row)

    if not rows:
        raise ValueError(f"No CSV files found in {raw_path}")

    points = pd.DataFrame(rows)
    points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
    points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
    points["height"] = pd.to_numeric(points[height_column], errors="coerce")
    points["lowland_fraction"] = pd.to_numeric(points["lowland_fraction"], errors="coerce")
    points = points.dropna(subset=["latitude", "longitude", "height"]).copy()

    sort_columns = [column for column in ["grid_row", "grid_col", "sample_id"] if column in points.columns]
    if sort_columns:
        points = points.sort_values(sort_columns)
    return points.reset_index(drop=True)


def show_3d_lowland(points: pd.DataFrame, height_column: str, lowland_threshold: float, azim: float, elev: float) -> None:
    """Display the 3D lowland plot interactively."""

    longitude = points["longitude"].to_numpy(dtype=float)
    latitude = points["latitude"].to_numpy(dtype=float)
    height = points["height"].to_numpy(dtype=float)
    lowland = points["lowland_fraction"].fillna(0).to_numpy(dtype=float)

    fig = plt.figure(figsize=(12, 8))
    axis = fig.add_subplot(111, projection="3d")

    if len(points) >= 3:
        surface = axis.plot_trisurf(
            longitude,
            latitude,
            height,
            cmap="terrain",
            linewidth=0.45,
            edgecolor="#666666",
            alpha=0.58,
            antialiased=True,
        )
        elevation_bar = fig.colorbar(surface, ax=axis, shrink=0.62, pad=0.10)
        elevation_bar.set_label("Height / elevation (m)")

    norm = Normalize(vmin=0.0, vmax=max(1.0, float(np.nanmax(lowland))))
    high_lowland = lowland >= lowland_threshold
    axis.scatter(
        longitude,
        latitude,
        height,
        c=cm.Blues(norm(lowland)),
        s=70 + (lowland * 120),
        edgecolors=np.where(high_lowland, "#B00020", "#222222"),
        linewidths=np.where(high_lowland, 1.3, 0.45),
        depthshade=True,
    )

    for _, row in points.iterrows():
        axis.text(
            float(row["longitude"]),
            float(row["latitude"]),
            float(row["height"]) + 0.5,
            str(row.get("sample_id", "")),
            fontsize=7,
        )

    lowland_bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="Blues"), ax=axis, shrink=0.62, pad=0.02)
    lowland_bar.set_label("Lowland fraction")

    axis.set_title("Sagil 3D Lowland and Elevation View", pad=18, fontsize=14, fontweight="bold")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_zlabel(f"Height from {height_column} (m)")
    axis.view_init(elev=elev, azim=azim)
    axis.grid(True, alpha=0.25)
    fig.text(
        0.5,
        0.03,
        f"Cells: {len(points)} | red edge = lowland_fraction >= {lowland_threshold:.2f}",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()


def main() -> None:
    """Load raw terrain rows and show the 3D plot."""

    args = parse_args()
    points = load_lowland_points(args.raw_dir, args.height_column)
    print(points[["sample_id", "latitude", "longitude", "height", "lowland_fraction"]].to_string(index=False))
    show_3d_lowland(points, args.height_column, args.lowland_threshold, args.azim, args.elev)


if __name__ == "__main__":
    main()
