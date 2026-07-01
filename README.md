# Sagil Monthly GEE Environmental Risk Forecasting

This repository builds a monthly Google Earth Engine time-series dataset for agricultural grid cells in Sagil, Tangkak, Johor, then prototypes an environmental anomaly index and one-month-ahead flood/drought proxy-risk forecast.

The learned score is an environmental anomaly and directional risk proxy derived from remote-sensing and reanalysis variables. It is not a direct disaster observation and must not be interpreted as a calibrated probability or percentage of agricultural land flooded or affected by drought.

Dataset repository:

```text
https://huggingface.co/datasets/Aki298/AML-Johor-Tangkak-Sagil-GEE-dataset
```

## Project Structure

```text
config/
  project_config.yaml

data/
  input/coordinates.csv          # tracked manual coordinate source
  metadata/sample_index.csv      # generated
  metadata/extraction_manifest.json
  raw/monthly/<sample_id>.csv    # generated per-cell monthly observations
  processed/processed_data.csv   # generated deterministic model table

notebooks/
  01_GEE_observation_extraction.ipynb
  02_feature_engineering.ipynb
  03_risk_scoring_and_forecasting.ipynb
  04_lstm_risk_forecasting.ipynb
  05_transformer_risk_forecasting.ipynb
  06_demo_coordinate_forecast.ipynb

src/
  gee_monthly_extraction.py
  monthly_feature_engineering.py
  risk_scoring.py
  transformer_forecasting.py
  demo_inference.py
  validation.py

scripts/
  predict_demo_coordinate.py
  repair_pytorch_cpu.py

artifacts/                      # generated models/scalers/thresholds
reports/                        # generated metrics/predictions
```

## Coordinate Input

Edit and track:

```text
data/input/coordinates.csv
```

Required columns:

```text
sample_id,grid_row,grid_col,latitude,longitude
```

Optional columns:

```text
study_area,cell_size_m,notes
```

Recommended sample IDs use stable grid names, for example `sagil_r00_c00`. Do not use latitude/longitude as primary filenames.

Each row is treated as the centroid of a 1,000 m x 1,000 m grid cell. The square is built in `EPSG:32648` using a 500 m half-width, then transformed to WGS84 for Earth Engine.

## Monthly GEE Dataset

The configured period is inclusive month starts:

```text
2021-01-01 through 2026-06-01
```

This shorter period reduces noise from older observations and creates 66 rows per coordinate. Each Earth Engine query uses the interval:

```text
[month_start, next_month_start)
```

The final month `2026-06-01` queries through `2026-07-01`.

Source datasets include Sentinel-2, Sentinel-1, CHIRPS, ERA5-Land, Dynamic World, SRTM, and JRC Global Surface Water. Monthly reducers follow the physical meaning of each variable: medians/means for indices and probabilities, sums for rainfall/runoff/evaporation totals, counts for dry/heavy-rain days, and static terrain/water values repeated per month.

## Pipeline

1. `01_GEE_observation_extraction.ipynb`
   - Reads `data/input/coordinates.csv`.
   - Validates coordinate uniqueness, month schedule, and 1 km cell geometry.
   - Writes per-cell raw monthly CSVs to `data/raw/monthly/`.
   - Writes `data/metadata/sample_index.csv` and `data/metadata/extraction_manifest.json`.

2. `02_feature_engineering.ipynb`
   - Reads `data/metadata/sample_index.csv` and raw monthly CSVs.
   - Validates monthly continuity and unique `(sample_id, month)` keys.
   - Keeps all raw observations and adds a reduced hydrology-focused set of calendar, lag, rolling, and difference features.
   - Writes `data/processed/processed_data.csv`.
   - Does not create learned anomaly scores, future targets, or model predictions.

3. `03_risk_scoring_and_forecasting.ipynb`
   - Loads `data/processed/processed_data.csv`.
   - Fits training-only imputers/scalers/reconstruction models.
   - Creates anomaly magnitude, flood/drought direction scores, severity thresholds, and one-month-ahead proxy targets.
   - Trains persistence, Ridge, and a reduced 100-tree Random Forest.
   - Saves all trained models, per-model prediction reports, and best-RMSE generic prediction reports.
   - Plots model metrics, test diagnostics, and holdout predictions.

4. `04_lstm_risk_forecasting.ipynb`
   - Reuses the same scored forecasting table and chronological splits.
   - Requires PyTorch and trains a real LSTM sequence regressor.
   - Attempts the local PyTorch CPU repair helper if PyTorch import fails, then raises clearly if LSTM cannot run.
   - Saves LSTM sequence artifacts for demo inference when this notebook is run.
   - Plots sequence training history, model comparison, test diagnostics, and holdout predictions.

5. `05_transformer_risk_forecasting.ipynb`
   - Reuses the same scored forecasting table and chronological splits.
   - Builds 12-month sequences and trains a compact PyTorch Transformer encoder.
   - Saves Transformer artifacts, predictions, metrics, and a model-comparison report.
   - Updates the generic best-model reports only when Transformer has the lowest test RMSE.
   - This is an advanced optional comparison model and may not outperform simpler models on the small current dataset.

6. `06_demo_coordinate_forecast.ipynb`
   - Runs the live or offline coordinate demo in a separate notebook.
   - Loads trained artifacts from notebooks 03, 04, and 05 without retraining.
   - Writes and visualizes `reports/demo_coordinate_prediction.csv`.

## Temporal Split

All learned transformations are fitted after splitting logic is defined. Forecast rows are split by `target_month`:

```text
train:   target_month <= 2025-05-01
test:    2025-06-01 <= target_month <= 2026-05-01
holdout: target_month == 2026-06-01
```

The observation period starts at `2021-01-01`. After the one-month target shift, supervised training uses source rows from `2021-01-01` onward and target months through `2025-05-01`. The holdout target month `2026-06-01` uses source features from `2026-05-01`. The June 2026 source row has no July 2026 target and is excluded from supervised evaluation.

## Live Coordinate Demo

Use `06_demo_coordinate_forecast.ipynb` or the demo script to predict `month+1` risk for a coordinate inside the selected Sagil square. The coordinate is treated as the centroid of a 1 km x 1 km cell. By default the workflow uses the latest complete month and downloads enough recent history to compute the reduced lag/rolling features.

```powershell
conda run -n aml python scripts/predict_demo_coordinate.py --latitude 2.31386663 --longitude 102.6555956
```

For an offline smoke test using an existing raw monthly CSV:

```powershell
conda run -n aml python scripts/predict_demo_coordinate.py --latitude 2.32289114 --longitude 102.6465942 --sample-id demo_sagil_1 --source-month 2026-05-01 --raw-csv data/raw/monthly/sagil_1.csv
```

The output is written to `reports/demo_coordinate_prediction.csv`.

When `artifacts/transformer_model_metadata.json` exists, the demo also attempts a Transformer prediction and uses it only if `artifacts/model_registry.json` marks Transformer as the best model.

## Environment Setup

```powershell
conda activate aml
python -m pip install -r requirements.txt
```

If PyTorch import fails on Windows, run the local CPU repair helper:

```powershell
python scripts/repair_pytorch_cpu.py
```

The helper verifies `import torch` first and only reinstalls CPU wheels when the import fails. It uses the official PyTorch CPU wheel index for the Windows/Pip/CPU install path.

Official PyTorch local install guidance: https://pytorch.org/get-started/locally/

For Earth Engine, set `GEE_PROJECT_ID` or create ignored local credentials:

```text
config/gee_credentials.json
```

with:

```json
{
  "project_id": "your-gee-project-id"
}
```

## Pull Dataset from Hugging Face

```powershell
conda run -n aml python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Aki298/AML-Johor-Tangkak-Sagil-GEE-dataset', repo_type='dataset', local_dir='data')"
```

Generated data, artifacts, and reports are intentionally ignored by Git except `data/input/coordinates.csv`.
