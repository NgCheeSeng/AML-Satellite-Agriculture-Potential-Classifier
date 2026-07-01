# Plan: Sagil Monthly GEE Environmental Risk Forecasting

## Objective

Replace the previous satellite-image classification workflow with a monthly, grid-based Google Earth Engine time-series project for agricultural land in Sagil, Tangkak, Johor. The project extracts environmental observations for manually supplied 1 km grid-cell centroids, builds deterministic time-series features, learns an unsupervised environmental anomaly index, and forecasts the next-month signed flood/drought proxy-risk score.

The score is a learned environmental anomaly proxy, not a calibrated flood or drought probability.

## Data Contract

Tracked input:

```text
data/input/coordinates.csv
```

Required columns:

```text
sample_id,grid_row,grid_col,latitude,longitude
```

Generated outputs:

```text
data/metadata/sample_index.csv
data/metadata/extraction_manifest.json
data/raw/monthly/<sample_id>.csv
data/processed/processed_data.csv
artifacts/*.joblib
artifacts/risk_thresholds.json
artifacts/feature_columns.json
artifacts/model_registry.json
artifacts/lstm_model.pt
artifacts/sequence_feature_imputer.joblib
artifacts/sequence_feature_scaler.joblib
artifacts/sequence_model_metadata.json
artifacts/transformer_model.pt
artifacts/transformer_feature_imputer.joblib
artifacts/transformer_feature_scaler.joblib
artifacts/transformer_model_metadata.json
reports/metrics.json
reports/*_predictions.csv
reports/lstm_test_predictions.csv
reports/lstm_holdout_predictions.csv
reports/model_comparison_with_lstm.csv
reports/demo_coordinate_prediction.csv
reports/transformer_metrics.json
reports/transformer_test_predictions.csv
reports/transformer_holdout_predictions.csv
reports/model_comparison_with_transformer.csv
```

`coordinates.csv` is the source of truth. Automatic 10 km x 10 km grid generation is out of scope for the first implementation.

## Monthly Extraction

Configured months are inclusive month starts from `2021-01-01` through `2026-06-01`, producing 66 months per coordinate. This shorter period reduces noise from older observations while keeping the recent hydrology and land-condition trend. Each extraction interval is `[month_start, next_month_start)`, so June 2026 queries through `2026-07-01`.

Each coordinate is converted into a 1,000 m x 1,000 m square cell using `EPSG:32648` with 500 m half-width, then transformed to WGS84 for Earth Engine. The expected projected area is approximately 1,000,000 m2.

Monthly raw observations include Sentinel-2 optical indices, Sentinel-1 SAR, CHIRPS rainfall totals/counts, ERA5-Land climate and water variables, Dynamic World probabilities, static terrain, historical maximum water extent, image counts, valid-pixel fractions, and missingness flags. Missing satellite values are not silently converted to zero.

## Notebook Pipeline

1. `01_GEE_observation_extraction.ipynb` validates coordinates, builds the monthly schedule, extracts GEE observations, writes one raw CSV per sample, and writes metadata.
2. `02_feature_engineering.ipynb` combines raw monthly CSVs, validates continuity, keeps all raw observations, creates a reduced hydrology-focused set of calendar/lag/rolling/difference features, and writes `processed_data.csv`. It does not create learned risk scores or future targets.
3. `03_risk_scoring_and_forecasting.ipynb` fits training-only imputation/scaling/reconstruction models, scores anomaly magnitude and flood/drought direction, creates one-month-ahead proxy targets, trains persistence/Ridge/Random Forest baselines, reports evaluation metrics, and saves all model artifacts/reports. Generic prediction reports use the lowest test-RMSE model.
4. `04_lstm_risk_forecasting.ipynb` requires PyTorch, preloads or repairs the local CPU PyTorch install before scoring, trains a real LSTM sequence regressor on the same scored forecasting table, and saves LSTM sequence artifacts.
5. `05_transformer_risk_forecasting.ipynb` trains a compact PyTorch Transformer encoder on the same 12-month sequence contract, saves Transformer-specific artifacts and reports, compares against existing models, and updates generic best-model reports only if Transformer has the lowest test RMSE.
6. `06_demo_coordinate_forecast.ipynb` runs live or offline coordinate inference separately from model training, loads saved artifacts without retraining, writes `reports/demo_coordinate_prediction.csv`, and visualizes the available per-model predictions.

## Modeling Contract

Unsupervised anomaly scoring is fitted only on training-period rows. Reconstruction baselines include PCA and an `MLPRegressor` trained as an autoencoder-style `X_scaled -> X_scaled` model. Reconstruction error is anomaly magnitude. Flood/drought direction is derived from standardized environmental direction indicators and is diagnostic, not calibrated.

## Proxy Score Calculation

The proxy score is calculated once after deterministic feature engineering and is then used as the target for all forecasting models. The forecasting models do not create different definitions of risk; they predict the same next-month signed proxy score using different model families.

### Training-fitted preprocessing

Only training-period rows are used to fit the imputer, scaler, PCA reconstruction model, MLP reconstruction model, and risk thresholds.

For each monthly row, selected environmental reconstruction variables are transformed as:

```text
X_raw
-> median imputer fitted on training rows
-> standard scaler fitted on training rows
-> X_scaled
```

### Anomaly magnitude

Two reconstruction errors are calculated:

```text
pca_anomaly_magnitude = mean((X_scaled - PCA_inverse(PCA_transform(X_scaled)))^2)
anomaly_magnitude     = mean((X_scaled - MLP_reconstruct(X_scaled))^2)
```

`pca_anomaly_magnitude` is kept as a diagnostic baseline. The current primary proxy-risk magnitude is `anomaly_magnitude` from the shallow `MLPRegressor` reconstruction model.

### Flood and drought direction

The reconstruction error is non-negative, so it only measures how unusual the environmental state is. Direction is assigned separately using environmental indicators.

Flood-direction score is the average of positive standardized flood indicators:

```text
positive rainfall_total_mm
positive heavy_rain_days_count
positive surface_runoff_total_m
positive soil_water_layer1_mean
positive water_probability_mean
positive flooded_vegetation_probability_mean
positive vh_mean_db
positive lowland_fraction
```

Drought-direction score is the average of drought indicators:

```text
negative rainfall_total_mm
positive dry_days_count
negative soil_water_layer1_mean
negative ndmi_mean
negative ndvi_mean
negative surface_runoff_total_m
```

Direction rule:

```text
if flood_direction_score > drought_direction_score:
    risk_direction = flood
elif drought_direction_score > flood_direction_score:
    risk_direction = drought
else:
    risk_direction = neutral
```

The signed proxy score is then:

```text
signed_risk_score = anomaly_magnitude * direction_sign

direction_sign = +1 for flood
                 -1 for drought
                  0 for neutral
```

Interpretation:

```text
positive signed_risk_score -> flood-directed environmental anomaly
negative signed_risk_score -> drought-directed environmental anomaly
near-zero score            -> normal or directionally weak anomaly
```

The score is not a calibrated probability, percentage flooded, or verified disaster label.

### Severity buckets

Severity thresholds are fitted only from training-period `anomaly_magnitude` values:

```text
moderate >= training 75th percentile
high     >= training 90th percentile
extreme  >= training 97.5th percentile
low      <  training 75th percentile
```

Severity uses anomaly magnitude, while flood/drought direction uses the sign of `signed_risk_score`.

### Forecasting target

Within each `sample_id`, the supervised target is generated by shifting the signed proxy score one month forward:

```text
target_month = month.shift(-1)
target_risk_score_t_plus_1 = signed_risk_score.shift(-1)
```

Rows without a next month are marked unavailable and excluded from supervised evaluation.

### Per-model proxy-score prediction

All forecasting models predict the same numerical target:

```text
features available at month t -> target_risk_score_t_plus_1
```

Model-specific prediction rules:

```text
Persistence:
  predicted_proxy_score(t+1) = signed_risk_score(t)

Ridge regression:
  predicted_proxy_score(t+1) = Ridge(X_t)

Random Forest:
  predicted_proxy_score(t+1) = RandomForestRegressor(X_t)

LSTM:
  predicted_proxy_score(t+1) = LSTM([X_{t-11}, ..., X_t])

Transformer:
  predicted_proxy_score(t+1) = TransformerEncoder([X_{t-11}, ..., X_t])
```

`X_t` and sequence inputs use leakage-safe numeric features only. Identifier columns, target columns, actual/predicted columns, `target_month`, and categorical direction/severity labels are excluded from model inputs.

The demo coordinate workflow loads all available model artifacts, writes each model prediction into `reports/demo_coordinate_prediction.csv`, and selects `best_model_predicted_proxy_score` according to `artifacts/model_registry.json`.

Forecast rows are split by `target_month`:

```text
train:   target_month <= 2025-05-01
test:    2025-06-01 <= target_month <= 2026-05-01
holdout: target_month == 2026-06-01
```

The observation period starts at `2021-01-01`. Because the target is shifted one month forward, training source rows start at `2021-01-01`, while the first available supervised target month is `2021-02-01`. The training period ends at target month `2025-05-01`.

Mandatory forecasting baselines are persistence and Ridge regression. The main tabular prototype is a reduced `RandomForestRegressor` with 100 trees. Optional sequence models include a real PyTorch LSTM and a compact PyTorch Transformer encoder for advanced comparison. Report MAE, median absolute error, RMSE, R2, bias, and flood/drought sign accuracy on the chronological test period. Save all trained models for the June 2026 and live-coordinate demonstrations. Sequence models are optional and may not outperform simpler models on the small current dataset.

Live coordinate demo: `06_demo_coordinate_forecast.ipynb` and `scripts/predict_demo_coordinate.py` treat a provided coordinate as a 1 km cell centroid, extract or load recent monthly observations, rebuild the reduced deterministic features, load saved artifacts, and write `reports/demo_coordinate_prediction.csv` for `source_month -> source_month + 1`.

If Transformer artifacts are present, the demo includes `transformer_predicted_proxy_score` and only selects it as the best output when `artifacts/model_registry.json` marks `transformer` as best.

PyTorch repair helper: `scripts/repair_pytorch_cpu.py` verifies `import torch` and reinstalls CPU wheels from the official PyTorch CPU wheel index only when the import fails.

Official PyTorch local install guidance: https://pytorch.org/get-started/locally/

## Validation

Validate notebook JSON, compile all `src/*.py`, ensure obsolete old-pipeline terms are removed from source/docs/notebooks, confirm month schedule length is 66, confirm unique coordinate IDs and grid cells, confirm raw CSV continuity and unique `(sample_id, month)`, confirm no learned or future columns exist in `processed_data.csv`, confirm model fitting uses training-period data only, confirm `target_risk_score_t_plus_1` is a one-month within-sample shift of `signed_risk_score`, and confirm every forecasting model predicts the same proxy-score target.

## Scope Boundaries

Included: monthly GEE extraction, deterministic feature engineering, unsupervised anomaly learning, flood/drought directional interpretation, one-month-ahead proxy-score forecasting, chronological evaluation.

Excluded: automatic grid generation, verified flood-event labels, crop-loss records, calibrated disaster probabilities, real-time forecasts beyond the monthly GEE demo, and production deployment.
