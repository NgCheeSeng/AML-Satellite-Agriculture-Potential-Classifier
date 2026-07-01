"""Transformer sequence forecasting helpers for monthly Sagil risk prediction."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransformerConfig:
    """Small Transformer settings for the current Sagil dataset size."""

    sequence_length: int = 12
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 42


def require_torch() -> Any:
    """Import torch or raise a clear runtime error."""

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for the Transformer notebook. Run: "
            "python scripts/repair_pytorch_cpu.py"
        ) from exc
    return torch


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sequence_dataset(
    table: pd.DataFrame,
    split_mask: pd.Series,
    sequence_length: int,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build fixed-length monthly sequences within each sample_id."""

    selected_indices = set(table.index[split_mask])
    x_values = []
    y_values = []
    metadata_rows = []
    for _, sample_data in table.groupby("sample_id", sort=False):
        sample_data = sample_data.sort_values("month")
        sample_indices = list(sample_data.index)
        for position, row_index in enumerate(sample_indices):
            if row_index not in selected_indices:
                continue
            start = position - sequence_length + 1
            if start < 0:
                continue
            window_indices = sample_indices[start:position + 1]
            window = table.loc[window_indices, feature_columns]
            if window.isna().any().any():
                continue
            x_values.append(window.to_numpy(dtype=np.float32))
            y_values.append(float(table.loc[row_index, "target_risk_score_t_plus_1"]))
            metadata_rows.append(table.loc[row_index, [
                "sample_id",
                "month",
                "target_month",
                "target_risk_score_t_plus_1",
                "risk_direction",
                "risk_severity",
            ]])
    x_array = np.asarray(x_values, dtype=np.float32)
    y_array = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
    metadata = pd.DataFrame(metadata_rows).reset_index(drop=True) if metadata_rows else pd.DataFrame()
    return x_array, y_array, metadata


def make_transformer_model(
    input_size: int,
    sequence_length: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
) -> Any:
    """Create a TransformerRiskRegressor after importing torch."""

    require_torch()
    return TransformerRiskRegressor(
        input_size=input_size,
        sequence_length=sequence_length,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


def _base_module():
    """Return torch.nn.Module if torch is available, else object for import-time safety."""

    try:
        import torch

        return torch.nn.Module
    except Exception:
        return object


class PositionalEncoding(_base_module()):
    """Sinusoidal positional encoding for monthly sequence order."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        torch = require_torch()
        nn = torch.nn
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerRiskRegressor(_base_module()):
    """Transformer encoder for sequence-to-one proxy-risk regression."""

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        torch = require_torch()
        nn = torch.nn
        super().__init__()
        self.input_size = int(input_size)
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_value = float(dropout)
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=sequence_length, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        projected = self.input_projection(x)
        encoded = self.encoder(self.positional_encoding(projected))
        return self.head(encoded[:, -1, :])


def train_transformer(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TransformerConfig,
) -> pd.DataFrame:
    """Train the Transformer with chronological tail validation and early stopping."""

    torch = require_torch()
    nn = torch.nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    split_index = max(1, int(len(x_train) * 0.85))
    if split_index >= len(x_train):
        split_index = len(x_train) - 1
    train_dataset = TensorDataset(
        torch.tensor(x_train[:split_index], dtype=torch.float32),
        torch.tensor(y_train[:split_index], dtype=torch.float32),
    )
    valid_dataset = TensorDataset(
        torch.tensor(x_train[split_index:], dtype=torch.float32),
        torch.tensor(y_train[split_index:], dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_valid_loss = float("inf")
    wait = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                valid_losses.append(float(loss_fn(model(x_batch), y_batch).detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        valid_loss = float(np.mean(valid_losses)) if valid_losses else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return pd.DataFrame(history)


def predict_transformer(model: Any, x_array: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Predict with a trained Transformer model."""

    if len(x_array) == 0:
        return np.array([])
    torch = require_torch()
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    loader = DataLoader(torch.tensor(x_array, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch.to(device)
            predictions.append(model(x_batch).detach().cpu().numpy().reshape(-1))
    model.to("cpu")
    return np.concatenate(predictions)


def prediction_frame(rows: pd.DataFrame, predicted: np.ndarray, split: str, model_name: str = "transformer") -> pd.DataFrame:
    """Build a standard sequence prediction report."""

    output = rows[["sample_id", "month", "target_month", "target_risk_score_t_plus_1", "risk_direction", "risk_severity"]].copy()
    output = output.rename(columns={"target_risk_score_t_plus_1": "actual_proxy_score"})
    output["predicted_proxy_score"] = predicted
    output["model"] = model_name
    output["split"] = split
    return output
