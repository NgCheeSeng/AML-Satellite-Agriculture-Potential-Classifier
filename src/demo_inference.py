"""Live coordinate inference for the Sagil monthly risk demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.gee_monthly_extraction import extract_cell_monthly_rows, initialize_earth_engine
from src.monthly_feature_engineering import add_deterministic_features
from src.risk_scoring import score_risk
from src.validation import (
    build_square_cell,
    load_project_config,
    next_month_start,
    path_from_config,
    project_path,
    validate_cell_area,
)


@dataclass(frozen=True)
class DemoCoordinateConfig:
    """Settings for one live coordinate forecast."""

    latitude: float
    longitude: float
    sample_id: str = "demo_coordinate"
    source_month: str | None = None
    history_months: int | None = None
    allow_incomplete_source_month: bool = False
    output_csv: str | Path = "reports/demo_coordinate_prediction.csv"
    config_path: str | Path = "config/project_config.yaml"
    raw_csv: str | Path | None = None


def predict_demo_coordinate(demo_config: DemoCoordinateConfig, ee_module: Any | None = None) -> pd.DataFrame:
    """Extract recent monthly rows for one coordinate and write one month+1 prediction."""

    config = load_project_config(demo_config.config_path)
    source_month, source_status = resolve_source_month(
        demo_config.source_month,
        allow_incomplete=demo_config.allow_incomplete_source_month,
    )
    history_months = int(demo_config.history_months or config.get("demo", {}).get("default_history_months", 13))
    if history_months < 2:
        raise ValueError("history_months must be at least 2")

    if demo_config.raw_csv is None:
        monthly_rows = extract_demo_monthly_rows(demo_config, source_month, history_months, config, ee_module=ee_module)
    else:
        monthly_rows = load_demo_monthly_rows(demo_config.raw_csv, source_month, history_months)
        if monthly_rows.empty:
            raise ValueError("No monthly rows available at or before the requested source_month")
        available_source = str(monthly_rows["month"].max())
        if available_source != source_month:
            source_month = available_source
            source_status = "fell_back_to_latest_available_raw_month"

    prediction = predict_from_monthly_rows(
        monthly_rows=monthly_rows,
        latitude=float(demo_config.latitude),
        longitude=float(demo_config.longitude),
        sample_id=demo_config.sample_id,
        source_month=source_month,
        source_status=source_status,
        config=config,
    )
    output_path = project_path(demo_config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_path, index=False)
    return prediction


def resolve_source_month(source_month: str | None, allow_incomplete: bool = False) -> tuple[str, str]:
    """Return a source month and whether a fallback was applied."""

    latest_complete = latest_complete_month()
    if source_month is None:
        return latest_complete, "latest_complete_month"
    parsed = pd.Timestamp(source_month).normalize()
    if parsed.day != 1:
        raise ValueError("source_month must be a month start: YYYY-MM-01")
    current_month_start = pd.Timestamp.today().normalize().replace(day=1)
    if parsed >= current_month_start and not allow_incomplete:
        return latest_complete, "fell_back_to_latest_complete_month"
    return parsed.date().isoformat(), "requested_source_month"


def latest_complete_month(reference_date: str | None = None) -> str:
    """Return the latest complete calendar month start."""

    today = pd.Timestamp(reference_date).normalize() if reference_date else pd.Timestamp.today().normalize()
    current_month_start = today.replace(day=1)
    return (current_month_start - pd.DateOffset(months=1)).date().isoformat()


def trailing_months(source_month: str, history_months: int) -> list[str]:
    """Build the inclusive history window ending at source_month."""

    end = pd.Timestamp(source_month).normalize()
    start = end - pd.DateOffset(months=history_months - 1)
    return [value.date().isoformat() for value in pd.date_range(start=start, end=end, freq="MS")]


def extract_demo_monthly_rows(
    demo_config: DemoCoordinateConfig,
    source_month: str,
    history_months: int,
    config: dict[str, Any],
    ee_module: Any | None = None,
) -> pd.DataFrame:
    """Download recent GEE rows for the demo coordinate."""

    grid_config = config.get("grid", {})
    cell = build_square_cell(
        float(demo_config.latitude),
        float(demo_config.longitude),
        float(grid_config.get("half_width_m", 500)),
        str(grid_config.get("projected_crs", "EPSG:32648")),
    )
    validate_cell_area(
        float(cell["cell_area_m2"]),
        float(grid_config.get("expected_area_m2", 1_000_000)),
        float(grid_config.get("area_tolerance_fraction", 0.05)),
    )
    coordinate = pd.Series({
        "sample_id": demo_config.sample_id,
        "grid_row": -1,
        "grid_col": -1,
        "latitude": float(demo_config.latitude),
        "longitude": float(demo_config.longitude),
    })
    ee = initialize_earth_engine(config, ee_module=ee_module)
    return extract_cell_monthly_rows(ee, coordinate, cell["wgs84_ring"], trailing_months(source_month, history_months), config)


def load_demo_monthly_rows(raw_csv: str | Path, source_month: str, history_months: int) -> pd.DataFrame:
    """Load recent rows from an existing raw CSV for offline demo testing."""

    data = pd.read_csv(project_path(raw_csv))
    data = data.copy()
    data["month"] = pd.to_datetime(data["month"]).dt.date.astype(str)
    data = data[pd.to_datetime(data["month"]) <= pd.Timestamp(source_month)]
    return data.sort_values("month").tail(history_months).reset_index(drop=True)


def predict_from_monthly_rows(
    monthly_rows: pd.DataFrame,
    latitude: float,
    longitude: float,
    sample_id: str,
    source_month: str,
    source_status: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Predict month+1 risk from already loaded monthly rows."""

    processed = add_deterministic_features(monthly_rows)
    artifacts = load_risk_artifacts(config)
    preload_torch_runtime_if_available(config)
    scored = score_risk(processed, artifacts, config, thresholds=artifacts["risk_thresholds"])
    source_rows = scored.loc[scored["month"] == source_month]
    if source_rows.empty:
        raise ValueError(f"No source row found for {source_month}")
    source_row = source_rows.tail(1).copy()

    feature_columns = artifacts["forecasting_feature_columns"]
    for column in feature_columns:
        if column not in source_row.columns:
            source_row[column] = np.nan
    x = source_row[feature_columns]

    predictions: dict[str, float] = {
        "persistence": float(source_row["signed_risk_score"].iloc[0]),
        "ridge": float(artifacts["ridge_model"].predict(x)[0]),
        "random_forest": float(artifacts["random_forest_model"].predict(x)[0]),
    }
    sequence_prediction = predict_sequence_model_if_available(scored, config)
    if sequence_prediction is not None:
        predictions["sequence_model"] = float(sequence_prediction)
        predictions["lstm"] = float(sequence_prediction)
    transformer_prediction = predict_transformer_model_if_available(scored, config)
    if transformer_prediction is not None:
        predictions["transformer"] = float(transformer_prediction)

    best_model = artifacts["model_registry"].get("best_model", "ridge")
    if best_model not in predictions:
        best_model = "ridge"
    best_score = predictions[best_model]
    target_month = next_month_start(source_month)

    row = {
        "sample_id": sample_id,
        "latitude": latitude,
        "longitude": longitude,
        "source_month": source_month,
        "target_month": target_month,
        "history_start_month": str(scored["month"].min()),
        "history_end_month": str(scored["month"].max()),
        "history_month_count": int(len(scored)),
        "source_month_status": source_status,
        "persistence_predicted_proxy_score": predictions.get("persistence"),
        "ridge_predicted_proxy_score": predictions.get("ridge"),
        "random_forest_predicted_proxy_score": predictions.get("random_forest"),
        "sequence_model_predicted_proxy_score": predictions.get("sequence_model"),
        "lstm_predicted_proxy_score": predictions.get("lstm"),
        "transformer_predicted_proxy_score": predictions.get("transformer"),
        "best_model": best_model,
        "best_model_predicted_proxy_score": best_score,
        "predicted_direction": direction_from_score(best_score),
        "predicted_severity": severity_from_score(best_score, artifacts["risk_thresholds"]),
    }
    return pd.DataFrame([row])


def load_risk_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    """Load trained risk-scoring and forecasting artifacts."""

    artifacts_dir = path_from_config(config, "paths", "artifacts_dir")
    feature_columns_path = artifacts_dir / "feature_columns.json"
    registry_path = artifacts_dir / "model_registry.json"
    thresholds_path = artifacts_dir / "risk_thresholds.json"
    required = [feature_columns_path, registry_path, thresholds_path]
    required += [artifacts_dir / name for name in [
        "feature_imputer.joblib",
        "feature_scaler.joblib",
        "pca_reconstruction_model.joblib",
        "mlp_reconstruction_model.joblib",
        "ridge_model.joblib",
        "random_forest_model.joblib",
    ]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trained artifacts. Run notebook 03 first. Missing: {missing[:5]}")

    feature_data = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    return {
        "features": feature_data["reconstruction_features"],
        "forecasting_feature_columns": feature_data["forecasting_feature_columns"],
        "imputer": joblib.load(artifacts_dir / "feature_imputer.joblib"),
        "scaler": joblib.load(artifacts_dir / "feature_scaler.joblib"),
        "pca": joblib.load(artifacts_dir / "pca_reconstruction_model.joblib"),
        "mlp": joblib.load(artifacts_dir / "mlp_reconstruction_model.joblib"),
        "ridge_model": joblib.load(artifacts_dir / "ridge_model.joblib"),
        "random_forest_model": joblib.load(artifacts_dir / "random_forest_model.joblib"),
        "risk_thresholds": json.loads(thresholds_path.read_text(encoding="utf-8")),
        "model_registry": json.loads(registry_path.read_text(encoding="utf-8")),
    }


def predict_sequence_model_if_available(scored: pd.DataFrame, config: dict[str, Any]) -> float | None:
    """Return an optional sequence-model prediction when notebook 04 artifacts exist."""

    artifacts_dir = path_from_config(config, "paths", "artifacts_dir")
    metadata_path = artifacts_dir / "sequence_model_metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns", [])
    sequence_length = int(metadata.get("sequence_length", 0))
    if not feature_columns or sequence_length <= 0 or len(scored) < sequence_length:
        return None
    rows = scored.sort_values("month").tail(sequence_length).copy()
    for column in feature_columns:
        if column not in rows.columns:
            rows[column] = np.nan
    imputer = joblib.load(artifacts_dir / metadata.get("imputer_artifact", "sequence_feature_imputer.joblib"))
    scaler = joblib.load(artifacts_dir / metadata.get("scaler_artifact", "sequence_feature_scaler.joblib"))
    x_scaled = scaler.transform(imputer.transform(rows[feature_columns]))

    model_type = metadata.get("model")
    if model_type != "lstm":
        return None
    return predict_lstm_sequence(artifacts_dir, metadata, x_scaled)


def preload_torch_runtime_if_available(config: dict[str, Any]) -> bool:
    """Import PyTorch before scoring when sequence artifacts are present."""

    artifacts_dir = path_from_config(config, "paths", "artifacts_dir")
    has_lstm = (artifacts_dir / "sequence_model_metadata.json").exists()
    has_transformer = (artifacts_dir / "transformer_model_metadata.json").exists()
    if not has_lstm and not has_transformer:
        return False
    try:
        import torch  # noqa: F401
        if has_transformer:
            import src.transformer_forecasting  # noqa: F401
    except Exception:
        return False
    return True


def predict_lstm_sequence(artifacts_dir: Path, metadata: dict[str, Any], x_scaled: np.ndarray) -> float | None:
    """Predict with a saved PyTorch LSTM if torch is available."""

    try:
        import torch
        from torch import nn
    except Exception:
        return None

    class LSTMRiskRegressor(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])

    model = LSTMRiskRegressor(
        input_size=int(metadata["input_size"]),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        dropout=float(metadata.get("dropout", 0.0)),
    )
    state = torch.load(artifacts_dir / metadata.get("model_artifact", "lstm_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x_scaled.reshape(1, x_scaled.shape[0], x_scaled.shape[1]), dtype=torch.float32)
        return float(model(tensor).detach().cpu().numpy().reshape(-1)[0])


def predict_transformer_model_if_available(scored: pd.DataFrame, config: dict[str, Any]) -> float | None:
    """Return an optional Transformer prediction when notebook 05 artifacts exist."""

    artifacts_dir = path_from_config(config, "paths", "artifacts_dir")
    metadata_path = artifacts_dir / "transformer_model_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        import torch
        from src.transformer_forecasting import make_transformer_model, predict_transformer
    except Exception:
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns", [])
    sequence_length = int(metadata.get("sequence_length", 0))
    if not feature_columns or sequence_length <= 0 or len(scored) < sequence_length:
        return None

    rows = scored.sort_values("month").tail(sequence_length).copy()
    for column in feature_columns:
        if column not in rows.columns:
            rows[column] = np.nan
    imputer = joblib.load(artifacts_dir / metadata.get("imputer_artifact", "transformer_feature_imputer.joblib"))
    scaler = joblib.load(artifacts_dir / metadata.get("scaler_artifact", "transformer_feature_scaler.joblib"))
    x_scaled = scaler.transform(imputer.transform(rows[feature_columns]))
    model = make_transformer_model(
        input_size=int(metadata["input_size"]),
        sequence_length=int(metadata["sequence_length"]),
        d_model=int(metadata["d_model"]),
        nhead=int(metadata["nhead"]),
        num_layers=int(metadata["num_layers"]),
        dim_feedforward=int(metadata["dim_feedforward"]),
        dropout=float(metadata["dropout"]),
    )
    state = torch.load(artifacts_dir / metadata.get("model_artifact", "transformer_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    return float(predict_transformer(model, x_scaled.reshape(1, x_scaled.shape[0], x_scaled.shape[1]), batch_size=1)[0])


def direction_from_score(value: float) -> str:
    """Convert signed proxy score to a direction label."""

    if value > 0:
        return "flood"
    if value < 0:
        return "drought"
    return "neutral"


def severity_from_score(value: float, thresholds: dict[str, float]) -> str:
    """Convert signed proxy score magnitude to a severity bucket."""

    magnitude = abs(float(value))
    if magnitude >= thresholds["extreme"]:
        return "extreme"
    if magnitude >= thresholds["high"]:
        return "high"
    if magnitude >= thresholds["moderate"]:
        return "moderate"
    return "low"
