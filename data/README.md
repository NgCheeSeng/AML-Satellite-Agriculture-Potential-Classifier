---
license: other
language:
- en
pretty_name: Sagil Monthly GEE Environmental Risk Dataset
size_categories:
- 1K<n<10K
task_categories:
- tabular-regression
- time-series-forecasting
tags:
- earth-engine
- remote-sensing
- environmental-monitoring
- flood-risk
- drought-risk
- agriculture
- malaysia
- time-series
---

# Sagil Monthly GEE Environmental Risk Dataset

This dataset contains monthly Google Earth Engine environmental observations and deterministic time-series features for agricultural grid cells in Sagil, Tangkak, Johor, Malaysia.

It supports a prototype task:

```text
monthly environmental features at month t
-> one-month-ahead signed flood/drought proxy-risk score at month t+1
```

The learned score used by the project is an environmental anomaly and directional risk proxy. It is not a verified flood event label, crop-loss label, or calibrated flood/drought probability.

## Dataset Summary

- Study area: Sagil, Tangkak, Johor, Malaysia
- Spatial unit: 1 km x 1 km grid-cell centroid
- Current valid grid cells: 49
- Monthly period: 2021-01-01 through 2026-06-01
- Months per valid grid cell: 66
- Processed rows: 3,234
- Data type: tabular monthly spatiotemporal environmental data
- Primary file for modeling: `processed/processed_data.csv`

## Repository Structure

```text
input/
  coordinates.csv

metadata/
  sample_index.csv
  extraction_manifest.json

raw/
  monthly/
    sagil_1.csv
    sagil_2.csv
    ...

processed/
  processed_data.csv
```

## File Descriptions

### `input/coordinates.csv`

Manual coordinate source for the Sagil grid cells.

Important columns:

```text
sample_id
grid_row
grid_col
latitude
longitude
study_area
cell_size_m
```

Each row represents the centroid of a 1 km x 1 km grid cell. `grid_row = 0` is the northernmost row and `grid_col = 0` is the westernmost column.

### `metadata/sample_index.csv`

Extraction lookup table for raw monthly files.

Important columns:

```text
sample_id
grid_row
grid_col
latitude
longitude
cell_area_m2
first_month
last_month
row_count
raw_csv
extraction_status
```

Use rows with `extraction_status = ok` for modeling.

### `metadata/extraction_manifest.json`

Extraction run metadata, including:

```text
extraction timestamp
study area
first and last month
month count
source datasets
thresholds
failed samples
```

### `raw/monthly/<sample_id>.csv`

One raw monthly observation file per valid grid cell. Each file contains 66 monthly rows.

Raw observation groups include:

- Identifiers: `sample_id`, `grid_row`, `grid_col`, `latitude`, `longitude`, `month`
- Sentinel-2 optical indices: `ndvi_mean`, `evi_mean`, `ndmi_mean`, `ndwi_mean`
- Sentinel-2 quality: `s2_image_count`, `s2_valid_pixel_fraction`
- Sentinel-1 SAR: `vv_mean_db`, `vh_mean_db`, `vv_minus_vh_db`, `vv_vh_ratio_linear`, `s1_image_count`
- CHIRPS rainfall: `rainfall_total_mm`, `rainfall_mean_daily_mm`, `rainfall_max_1day_mm`, `dry_days_count`, `heavy_rain_days_count`
- ERA5-Land climate and water variables: `temperature_2m_mean_c`, `relative_humidity_mean_pct`, `soil_water_layer1_mean`, `surface_runoff_total_m`, `evaporation_total_m`
- Dynamic World probabilities: `water_probability_mean`, `flooded_vegetation_probability_mean`, `built_probability_mean`, `dynamicworld_image_count`
- Terrain and historical extent: `elevation_mean_m`, `elevation_min_m`, `elevation_max_m`, `slope_mean_deg`, `lowland_fraction`, `max_water_extent_fraction`
- Missingness flags: `optical_missing_flag`, `sar_missing_flag`, `climate_missing_flag`, `any_missing_flag`

### `processed/processed_data.csv`

Combined model-ready deterministic feature table.

It contains:

- all raw monthly observation columns
- calendar features: `year`, `month_number`, `month_sin`, `month_cos`
- lag features for selected hydrology, vegetation, SAR, and water variables
- rolling mean and rolling sum features
- month-to-month difference features

It intentionally does not contain:

- learned anomaly scores
- learned risk scores
- one-month-ahead target columns
- model predictions

Those values are generated in the modeling pipeline so that training-only preprocessing and target construction remain leakage-controlled.

## Source Datasets

The dataset is derived through Google Earth Engine using:

- Sentinel-2 Surface Reflectance Harmonized
- Sentinel-1 GRD
- CHIRPS Daily Rainfall
- ERA5-Land Daily Aggregates
- Dynamic World V1
- SRTM elevation
- JRC Global Surface Water maximum extent

Please consult the original providers' terms for downstream use and redistribution requirements.

## Intended Use

This dataset is intended for:

- academic machine learning experiments
- environmental time-series forecasting prototypes
- flood/drought proxy-risk modeling
- remote-sensing feature engineering demonstrations
- chronological train/test/holdout evaluation

The matching project code constructs a learned signed risk proxy and trains:

- persistence baseline
- Ridge regression
- Random Forest regressor
- LSTM sequence model
- Transformer encoder sequence model

## Not Intended For

This dataset should not be used as:

- an official flood warning dataset
- a calibrated disaster-probability dataset
- verified crop-loss or damage labels
- a substitute for field survey, hydrological model output, river gauge data, or official disaster records

## Risk Proxy Method

The project builds the target outside `processed_data.csv`.

For each grid cell `i` and month `t`, selected environmental variables form:

```text
x_i,t
```

Training-fitted preprocessing:

```text
z_i,t = StandardScaler(MedianImputer(x_i,t))
```

Anomaly magnitude:

```text
anomaly_magnitude = mean((z_i,t - MLP_reconstruct(z_i,t))^2)
```

Flood/drought direction is estimated from standardized environmental indicators:

```text
flood_direction_score = average of positive wetness and flood indicators
drought_direction_score = average of dry and vegetation-stress indicators
```

Signed proxy score:

```text
signed_risk_score = anomaly_magnitude * direction_sign
```

where:

```text
direction_sign = +1 for flood-directed anomaly
direction_sign = -1 for drought-directed anomaly
direction_sign = 0 for neutral or ambiguous anomaly
```

Forecasting target:

```text
target_risk_score_t_plus_1 = signed_risk_score at month t+1
```

## Temporal Split Used In The Project

The project code uses target-month splitting:

```text
train:   target_month <= 2025-05-01
test:    2025-06-01 <= target_month <= 2026-05-01
holdout: target_month == 2026-06-01
```

The June 2026 source row has no July 2026 target and is excluded from supervised evaluation.

## Loading Example

```python
from huggingface_hub import snapshot_download
import pandas as pd

local_dir = snapshot_download(
    repo_id="Aki298/AML-Johor-Tangkak-Sagil-GEE-dataset",
    repo_type="dataset",
    local_dir="data",
)

processed = pd.read_csv("data/processed/processed_data.csv")
coordinates = pd.read_csv("data/input/coordinates.csv")
```

## Limitations

- The target risk score is self-supervised and proxy-based, not field-verified.
- The current release covers one agricultural study area in Sagil, Johor.
- The effective independent sample size is smaller than the row count because neighboring grid cells and adjacent months are correlated.
- Next-month flood/drought behavior may depend on future rainfall and hydrological conditions that are not known at month `t`.
- Model results should be interpreted as prototype environmental-risk forecasting, not operational disaster forecasting.

## Citation

If using this dataset, cite the dataset repository and acknowledge the original Earth observation and climate data providers used through Google Earth Engine.

