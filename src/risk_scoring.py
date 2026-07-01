"""Prototype risk scoring and one-month-ahead forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.validation import load_project_config, path_from_config, write_json

RECONSTRUCTION_FEATURES = [
    "ndvi_mean", "evi_mean", "ndmi_mean", "ndwi_mean",
    "vv_mean_db", "vh_mean_db", "vv_minus_vh_db", "vv_vh_ratio_linear",
    "rainfall_total_mm", "rainfall_mean_daily_mm", "rainfall_max_1day_mm", "dry_days_count", "heavy_rain_days_count",
    "temperature_2m_mean_c", "relative_humidity_mean_pct", "soil_water_layer1_mean", "surface_runoff_total_m", "evaporation_total_m",
    "water_probability_mean", "flooded_vegetation_probability_mean", "built_probability_mean",
]
IDENTIFIER_COLUMNS = {
    "sample_id", "grid_row", "grid_col", "latitude", "longitude", "month", "target_month", "study_area", "notes",
}
TARGET_PREFIXES = ("target_", "predicted_", "actual_")
SUPERVISED_MODEL_NAMES = ["persistence", "ridge", "random_forest"]


def run_risk_scoring_and_forecasting(config_path: str | Path = "config/project_config.yaml") -> dict[str, Any]:
    """Run the prototype scoring and forecasting workflow end to end."""

    config, scored, forecasting_table, splits, artifacts = prepare_scored_forecasting_table(config_path)
    path_from_config(config, "paths", "artifacts_dir").mkdir(parents=True, exist_ok=True)
    path_from_config(config, "paths", "reports_dir").mkdir(parents=True, exist_ok=True)

    model_result = train_forecasting_models(forecasting_table, splits, config)
    save_artifacts(artifacts, model_result, config)
    save_reports(forecasting_table, model_result, config)
    return {
        "scored_rows": int(len(scored)),
        "forecasting_rows": int(len(forecasting_table)),
        "best_model": model_result["best_model"],
        "metrics": model_result["metrics"],
    }


def prepare_scored_forecasting_table(
    config_path: str | Path = "config/project_config.yaml",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, Any]]:
    """Create scored rows, one-month targets, and chronological split masks."""

    config = load_project_config(config_path)
    processed_path = path_from_config(config, "paths", "processed_data_csv")
    data = pd.read_csv(processed_path)
    data["month"] = pd.to_datetime(data["month"]).dt.date.astype(str)

    train_end = pd.Timestamp(config["modeling"]["train_target_end"])
    scoring_train = data[pd.to_datetime(data["month"]) <= train_end].copy()
    artifacts = fit_reconstruction_models(scoring_train, config)
    scored = score_risk(data, artifacts, config)
    artifacts["risk_thresholds"] = scored.attrs.get("risk_thresholds", {})
    forecasting_table = build_forecasting_table(scored)
    forecasting_table.attrs["risk_thresholds"] = artifacts["risk_thresholds"]
    splits = split_forecasting_rows(forecasting_table, config)
    return config, scored, forecasting_table, splits, artifacts


def fit_reconstruction_models(train_data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Fit training-only imputer, scaler, PCA, and MLP reconstruction models."""

    features = available_reconstruction_features(train_data)
    if not features:
        raise ValueError("No reconstruction features are available in processed_data.csv")
    x_raw = train_data[features]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imputed = imputer.fit_transform(x_raw)
    x_scaled = scaler.fit_transform(x_imputed)

    modeling = config.get("modeling", {})
    pca = PCA(n_components=modeling.get("pca_components", 0.95), random_state=int(modeling.get("random_state", 42)))
    pca.fit(x_scaled)

    hidden_layers = tuple(int(value) for value in modeling.get("mlp_hidden_layer_sizes", [8]))
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        random_state=int(modeling.get("random_state", 42)),
        max_iter=int(modeling.get("mlp_max_iter", 500)),
        early_stopping=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(x_scaled, x_scaled)

    return {
        "features": features,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "mlp": mlp,
    }


def score_risk(
    data: pd.DataFrame,
    artifacts: dict[str, Any],
    config: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Score anomaly magnitude, direction, and severity for every row."""

    scored = data.copy()
    features = artifacts["features"]
    x_scaled = transform_reconstruction_features(scored, artifacts)

    pca = artifacts["pca"]
    x_pca = pca.transform(x_scaled)
    x_pca_reconstructed = pca.inverse_transform(x_pca)
    scored["pca_anomaly_magnitude"] = np.mean(np.square(x_scaled - x_pca_reconstructed), axis=1)

    mlp = artifacts["mlp"]
    x_mlp_reconstructed = mlp.predict(x_scaled)
    scored["anomaly_magnitude"] = np.mean(np.square(x_scaled - x_mlp_reconstructed), axis=1)

    z = pd.DataFrame(x_scaled, columns=features, index=scored.index)
    flood_score, drought_score = direction_scores(scored, z)
    scored["flood_direction_score"] = flood_score
    scored["drought_direction_score"] = drought_score
    scored["risk_direction"] = np.select(
        [flood_score > drought_score, drought_score > flood_score],
        ["flood", "drought"],
        default="neutral",
    )
    direction_sign = np.select(
        [scored["risk_direction"] == "flood", scored["risk_direction"] == "drought"],
        [1.0, -1.0],
        default=0.0,
    )
    scored["signed_risk_score"] = scored["anomaly_magnitude"] * direction_sign

    fitted_thresholds = thresholds or fit_thresholds(scored, config)
    scored["risk_severity"] = apply_thresholds(scored["anomaly_magnitude"], fitted_thresholds)
    scored.attrs["risk_thresholds"] = fitted_thresholds
    return scored


def transform_reconstruction_features(data: pd.DataFrame, artifacts: dict[str, Any]) -> np.ndarray:
    """Apply the training-fitted imputer and scaler."""

    x_imputed = artifacts["imputer"].transform(data[artifacts["features"]])
    return artifacts["scaler"].transform(x_imputed)


def available_reconstruction_features(data: pd.DataFrame) -> list[str]:
    """Return reconstruction features present in the dataframe."""

    return [column for column in RECONSTRUCTION_FEATURES if column in data.columns and data[column].notna().any()]


def direction_scores(data: pd.DataFrame, z: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Create diagnostic flood and drought direction scores from standardized variables."""

    lowland = pd.to_numeric(data.get("lowland_fraction", pd.Series(0.0, index=data.index)), errors="coerce")
    flood_terms = [
        _positive(z, "rainfall_total_mm"),
        _positive(z, "heavy_rain_days_count"),
        _positive(z, "surface_runoff_total_m"),
        _positive(z, "soil_water_layer1_mean"),
        _positive(z, "water_probability_mean"),
        _positive(z, "flooded_vegetation_probability_mean"),
        _positive(z, "vh_mean_db"),
        lowland.clip(lower=0).fillna(0.0),
    ]
    drought_terms = [
        _negative(z, "rainfall_total_mm"),
        _positive(z, "dry_days_count"),
        _negative(z, "soil_water_layer1_mean"),
        _negative(z, "ndmi_mean"),
        _negative(z, "ndvi_mean"),
        _negative(z, "surface_runoff_total_m"),
    ]
    return _average_terms(flood_terms, data.index), _average_terms(drought_terms, data.index)


def fit_thresholds(scored: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    """Fit severity thresholds using training-period anomaly magnitudes only."""

    train_end = pd.Timestamp(config["modeling"]["train_target_end"])
    train_scores = scored.loc[pd.to_datetime(scored["month"]) <= train_end, "anomaly_magnitude"].dropna()
    if train_scores.empty:
        raise ValueError("Cannot fit risk thresholds without training-period anomaly scores")
    p75, p90, p975 = np.percentile(train_scores, config["modeling"].get("severity_percentiles", [75, 90, 97.5]))
    return {"moderate": float(p75), "high": float(p90), "extreme": float(p975)}


def apply_thresholds(values: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    """Convert anomaly magnitudes into relative severity buckets."""

    return pd.Series(
        np.select(
            [values >= thresholds["extreme"], values >= thresholds["high"], values >= thresholds["moderate"]],
            ["extreme", "high", "moderate"],
            default="low",
        ),
        index=values.index,
    )


def build_forecasting_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Create one-month-ahead target columns within each sample."""

    table = scored.sort_values(["sample_id", "month"]).copy()
    group = table.groupby("sample_id", group_keys=False)
    table["target_month"] = group["month"].shift(-1)
    table["target_risk_score_t_plus_1"] = group["signed_risk_score"].shift(-1)
    table["target_anomaly_magnitude_t_plus_1"] = group["anomaly_magnitude"].shift(-1)
    table["target_risk_direction_t_plus_1"] = group["risk_direction"].shift(-1)
    table["target_available_flag"] = table["target_month"].notna().astype(int)
    return table


def split_forecasting_rows(table: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.Series]:
    """Create non-overlapping chronological split masks based on target_month."""

    target_month = pd.to_datetime(table["target_month"])
    modeling = config["modeling"]
    train = target_month <= pd.Timestamp(modeling["train_target_end"])
    test = (target_month >= pd.Timestamp(modeling["test_target_start"])) & (target_month <= pd.Timestamp(modeling["test_target_end"]))
    holdout = target_month == pd.Timestamp(modeling["holdout_target_month"])
    available = table["target_available_flag"] == 1
    return {"train": train & available, "test": test & available, "holdout": holdout & available}


def train_forecasting_models(table: pd.DataFrame, splits: dict[str, pd.Series], config: dict[str, Any]) -> dict[str, Any]:
    """Train persistence, Ridge, and Random Forest forecasting models."""

    train = table.loc[splits["train"]].copy()
    test = table.loc[splits["test"]].copy()
    holdout = table.loc[splits["holdout"]].copy()
    feature_columns = [
        column for column in select_forecasting_feature_columns(table)
        if column in train.columns and train[column].notna().any()
    ]
    y_train = train["target_risk_score_t_plus_1"]
    y_test = test["target_risk_score_t_plus_1"]

    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=int(config["modeling"].get("random_forest_estimators", 100)),
            random_state=int(config["modeling"].get("random_state", 42)),
            n_jobs=1,
        )),
    ])

    ridge.fit(train[feature_columns], y_train)
    rf.fit(train[feature_columns], y_train)

    test_predictions = {
        "persistence": test["signed_risk_score"].to_numpy(),
        "ridge": ridge.predict(test[feature_columns]),
        "random_forest": rf.predict(test[feature_columns]),
    }
    holdout_predictions = {
        "persistence": holdout["signed_risk_score"].to_numpy() if not holdout.empty else np.array([]),
        "ridge": ridge.predict(holdout[feature_columns]) if not holdout.empty else np.array([]),
        "random_forest": rf.predict(holdout[feature_columns]) if not holdout.empty else np.array([]),
    }
    metrics = {name: regression_metrics(y_test, values) for name, values in test_predictions.items()}
    best_model = min(metrics, key=lambda name: metrics[name]["rmse"])

    test_prediction_frames = {
        name: prediction_frame(test, predictions, "test", model_name=name)
        for name, predictions in test_predictions.items()
    }
    holdout_prediction_frames = {
        name: prediction_frame(holdout, predictions, "holdout", model_name=name)
        for name, predictions in holdout_predictions.items()
    }

    return {
        "feature_columns": feature_columns,
        "ridge_model": ridge,
        "random_forest_model": rf,
        "metrics": metrics,
        "best_model": best_model,
        "test_predictions": test_prediction_frames[best_model],
        "holdout_predictions": holdout_prediction_frames[best_model],
        "test_prediction_frames": test_prediction_frames,
        "holdout_prediction_frames": holdout_prediction_frames,
        "demo_2026_06_predictions": wide_prediction_frame(holdout, holdout_predictions, best_model),
    }


def select_forecasting_feature_columns(table: pd.DataFrame) -> list[str]:
    """Select numeric, current-month model features without target leakage."""

    columns = []
    for column in table.columns:
        if column in IDENTIFIER_COLUMNS or column.startswith(TARGET_PREFIXES):
            continue
        if column in {"risk_direction", "risk_severity"}:
            continue
        if pd.api.types.is_numeric_dtype(table[column]):
            columns.append(column)
    return columns


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    residual = predicted_array - actual_array
    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "median_ae": float(median_absolute_error(actual_array, predicted_array)),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "r2": float(r2_score(actual_array, predicted_array)),
        "bias": float(np.mean(residual)),
        "direction_accuracy": float(direction_accuracy(actual_array, predicted_array)),
    }


def direction_accuracy(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Return sign agreement for flood-positive / drought-negative proxy scores."""

    actual_sign = np.sign(np.asarray(actual, dtype=float))
    predicted_sign = np.sign(np.asarray(predicted, dtype=float))
    valid = actual_sign != 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(actual_sign[valid] == predicted_sign[valid]))


def metrics_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert nested metric dictionaries to a sorted dataframe."""

    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(metrics, orient="index").reset_index(names="model").sort_values("rmse")


def prediction_frame(rows: pd.DataFrame, predicted: np.ndarray, split: str, model_name: str) -> pd.DataFrame:
    """Build a standard prediction output table."""

    output = rows[["sample_id", "month", "target_month", "target_risk_score_t_plus_1", "risk_direction", "risk_severity"]].copy()
    output = output.rename(columns={"target_risk_score_t_plus_1": "actual_proxy_score"})
    output["predicted_proxy_score"] = predicted
    output["model"] = model_name
    output["split"] = split
    return output


def wide_prediction_frame(rows: pd.DataFrame, predictions: dict[str, np.ndarray], best_model: str) -> pd.DataFrame:
    """Build a one-row-per-sample table with all model predictions."""

    base = rows[["sample_id", "month", "target_month", "target_risk_score_t_plus_1", "risk_direction", "risk_severity"]].copy()
    base = base.rename(columns={"target_risk_score_t_plus_1": "actual_proxy_score"})
    for name, values in predictions.items():
        base[f"{name}_predicted_proxy_score"] = values
    base["best_model"] = best_model
    base["best_model_predicted_proxy_score"] = predictions.get(best_model, np.array([]))
    base["split"] = "holdout"
    return base


def save_artifacts(artifacts: dict[str, Any], model_result: dict[str, Any], config: dict[str, Any]) -> None:
    """Persist fitted prototype models, feature lists, and thresholds."""

    artifacts_dir = path_from_config(config, "paths", "artifacts_dir")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts["imputer"], artifacts_dir / "feature_imputer.joblib")
    joblib.dump(artifacts["scaler"], artifacts_dir / "feature_scaler.joblib")
    joblib.dump(artifacts["pca"], artifacts_dir / "pca_reconstruction_model.joblib")
    joblib.dump(artifacts["mlp"], artifacts_dir / "mlp_reconstruction_model.joblib")
    joblib.dump(model_result["ridge_model"], artifacts_dir / "ridge_model.joblib")
    joblib.dump(model_result["random_forest_model"], artifacts_dir / "random_forest_model.joblib")

    thresholds = artifacts.get("risk_thresholds")
    if thresholds:
        write_json(artifacts_dir / "risk_thresholds.json", thresholds)
    write_json(artifacts_dir / "feature_columns.json", {
        "forecasting_feature_columns": model_result["feature_columns"],
        "reconstruction_features": artifacts["features"],
    })
    write_json(artifacts_dir / "model_registry.json", {
        "best_model": model_result["best_model"],
        "supervised_models": SUPERVISED_MODEL_NAMES,
        "artifacts": {
            "feature_imputer": "artifacts/feature_imputer.joblib",
            "feature_scaler": "artifacts/feature_scaler.joblib",
            "pca_reconstruction_model": "artifacts/pca_reconstruction_model.joblib",
            "mlp_reconstruction_model": "artifacts/mlp_reconstruction_model.joblib",
            "ridge_model": "artifacts/ridge_model.joblib",
            "random_forest_model": "artifacts/random_forest_model.joblib",
            "risk_thresholds": "artifacts/risk_thresholds.json",
            "feature_columns": "artifacts/feature_columns.json",
        },
    })


def save_reports(table: pd.DataFrame, model_result: dict[str, Any], config: dict[str, Any]) -> None:
    """Save metrics and prediction reports."""

    reports_dir = path_from_config(config, "paths", "reports_dir")
    artifacts_dir = path_from_config(config, "paths", "artifacts_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "metrics.json", model_result["metrics"])
    model_result["test_predictions"].to_csv(reports_dir / "test_predictions.csv", index=False)
    model_result["holdout_predictions"].to_csv(reports_dir / "holdout_predictions.csv", index=False)
    model_result["demo_2026_06_predictions"].to_csv(reports_dir / "demo_2026_06_predictions.csv", index=False)

    for name, frame in model_result["test_prediction_frames"].items():
        frame.to_csv(reports_dir / f"{name}_test_predictions.csv", index=False)
    for name, frame in model_result["holdout_prediction_frames"].items():
        frame.to_csv(reports_dir / f"{name}_holdout_predictions.csv", index=False)

    thresholds = table.attrs.get("risk_thresholds")
    if thresholds:
        write_json(artifacts_dir / "risk_thresholds.json", thresholds)


def _positive(z: pd.DataFrame, column: str) -> pd.Series:
    if column not in z.columns:
        return pd.Series(0.0, index=z.index)
    return z[column].clip(lower=0).fillna(0.0)


def _negative(z: pd.DataFrame, column: str) -> pd.Series:
    if column not in z.columns:
        return pd.Series(0.0, index=z.index)
    return (-z[column]).clip(lower=0).fillna(0.0)


def _average_terms(terms: list[pd.Series], index: pd.Index) -> pd.Series:
    if not terms:
        return pd.Series(0.0, index=index)
    aligned = [term.reindex(index).fillna(0.0) for term in terms]
    return sum(aligned) / len(aligned)
