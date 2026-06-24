"""Data readers for image sequence and final model input notebooks.

These helpers intentionally do not train models. They only build input tables and
merge existing per-sample outputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.gee_features import (
    load_per_sample_features,
    load_per_sample_targets,
    load_sample_index,
    project_relative_path,
    project_root_from_sample_index,
    resolve_project_path,
)

IMAGE_RESULT_COLUMNS = [
    "sample_id",
    "label",
    "frame_index",
    "acquisition_date",
    "urban_growth_probability",
]


def build_image_timeseries_input_index(
    sample_index_csv: str | Path = "data/processed/sample_index.csv",
) -> pd.DataFrame:
    """Build one image-row record per sample frame for the image model."""

    sample_index = load_sample_index(sample_index_csv)
    project_root = sample_index.attrs.get(
        "project_root",
        project_root_from_sample_index(sample_index_csv),
    )
    rows: list[dict[str, object]] = []
    for _, sample in sample_index.iterrows():
        frame_metadata_path = resolve_project_path(sample["frame_metadata_csv"], project_root)
        processed_dir = resolve_project_path(sample["processed_dir"], project_root)
        if not frame_metadata_path.exists():
            continue
        frame_metadata = pd.read_csv(frame_metadata_path)
        for _, frame in frame_metadata.iterrows():
            image_path = processed_dir / str(frame["image_file"])
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "label": sample["label"],
                    "latitude": sample["latitude"],
                    "longitude": sample["longitude"],
                    "frame_index": int(frame["frame_index"]),
                    "acquisition_date": frame["acquisition_date"],
                    "image_path": project_relative_path(image_path, project_root),
                    "image_exists": image_path.exists(),
                    "width_px": frame.get("width_px"),
                    "height_px": frame.get("height_px"),
                }
            )
    return pd.DataFrame(rows)


def load_image_timeseries_results(
    path: str | Path = "data/processed/image_timeseries_results.csv",
) -> pd.DataFrame:
    """Load optional image-model predictions with the expected schema."""

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=IMAGE_RESULT_COLUMNS)
    return pd.read_csv(path)


def load_model_training_inputs(
    sample_index_csv: str | Path = "data/processed/sample_index.csv",
    image_results_csv: str | Path = "data/processed/image_timeseries_results.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load GEE features, targets, image results, and their merged table."""

    features = load_per_sample_features(sample_index_csv)
    targets = load_per_sample_targets(sample_index_csv)
    image_results = load_image_timeseries_results(image_results_csv)
    merged = merge_features_targets_image_results(features, targets, image_results)
    return features, targets, image_results, merged


def merge_features_targets_image_results(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    image_results: pd.DataFrame,
) -> pd.DataFrame:
    """Merge model input tables on sample, label, frame, and date keys."""

    merge_keys = ["sample_id", "label", "frame_index", "acquisition_date"]
    merged = features.merge(targets, on=merge_keys, how="left", validate="one_to_one")
    if not image_results.empty:
        available_keys = [key for key in merge_keys if key in image_results.columns]
        image_columns = available_keys + [
            column for column in image_results.columns if column not in available_keys
        ]
        merged = merged.merge(
            image_results[image_columns],
            on=available_keys,
            how="left",
            validate="one_to_one",
        )
    return merged