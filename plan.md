# Concise Project Plan

## Project Title

**Satellite Image Time-Series Based Sustainable Agriculture Land Prediction**

## 1. Project Objective

This project aims to predict whether agricultural land remains sustainable for plantation using Sentinel-2 satellite image time series, elevation data, weather/rainfall indicators, vegetation indices, moisture indicators, and surrounding land-growth features.

The final output is a 3-class sustainable agriculture prediction:

1. Low
2. Moderate
3. High

The target application is to evaluate whether a new empty land area may be suitable for future durian/agricultural development.

### Current Implemented Data Pipeline

The current raw-data workflow uses MP4 + timeline TXT pairs:

```text
raw_to_be_processed/
  <latitude>_<longitude>_<label>.mp4
  <latitude>_<longitude>.txt
```

The fallback TXT name `<latitude>_<longitude>_<label>.txt` is also accepted for
current samples.

The TXT file contains one date per line. Line 1 maps to MP4 timestamp `0s`,
line 2 maps to `1s`, and so on. The preprocessing pipeline crops 5% from each
frame border and saves labelled processed frames here:

```text
data/raw/<label>/<latitude>_<longitude>/
data/processed/<label>/<latitude>_<longitude>/
```

The implemented notebook is:

```text
notebooks/01_video_to_cropped_frames.ipynb
```

GEE/environmental features are extracted later and saved into the same processed
sample folder as separate `gee_features.csv` and `gee_targets.csv` files.

---

## 2. Project Pipeline

```text
Data Collection
→ Image Preprocessing
→ Feature Engineering
→ Model Training
→ Model Evaluation
→ Demonstration on New Region
```

---

## 3. Input and Output

### Input

A **Satellite Image Time Series (SITS)** consisting of a variable number of Sentinel-2 timeframes for each selected region.

Each region includes:

* Sentinel-2 L2A agriculture-layer image sequence
* Center coordinate: latitude and longitude
* Elevation height data
* NDVI / NDWI / moisture stress / temperature / rainfall / SAR-related features
* Surrounding area indicators such as nearby urban growth, road access, flood-prone condition, and population context

### Output

A predicted sustainable agriculture class:

```text
Low / Moderate / High
```

---

## 4. Dataset Design

### 4.1 Satellite Image Time-Series Dataset

Satellite images will be obtained from **Copernicus Browser** using Sentinel-2 L2A imagery.

Target settings:

```text
Dataset: Sentinel-2 L2A
Date range: 2015 to latest available year
Cloud coverage: below 20%
Layer: Agriculture visualization layer
Timeframes per region: variable; one extracted frame per TXT timeline date
Number of labelled regions: 30–40 total
Training data: around 20–30 regions
Testing data: around 10 regions
```

The agriculture visualization layer is selected because it gives stronger visual contrast between vegetation, forest, bare land, built-up areas, and cloud-covered regions compared with NDVI/EVI visualization, which usually compresses information into a limited white-to-green scale.

### 4.2 Data Limitation

Since the project currently cannot download original GeoTIFF images, each satellite image time series will first be downloaded as an MP4 from Copernicus Browser with a matching TXT timeline file. The MP4 will then be converted into cropped PNG frames using Python.

The images are lossy video frames, so they will mainly be used for **visual temporal-pattern learning**, not precise spectral measurement. For accurate numerical features such as NDVI, NDWI, rainfall, temperature, and SAR indicators, Google Earth Engine or related APIs will be used later in `02_extract_gee_features.ipynb`.

All image frames must follow the same:

```text
map scale
center coordinate
crop size
visual layer
time order
```

---

## 5. Dataset Folder Structure

```text
aml_durian_agri_potential/
|
|-- raw_to_be_processed/
|   |-- <latitude>_<longitude>_<label>.mp4
|   |-- <latitude>_<longitude>.txt
|   `-- <latitude>_<longitude>_<label>.txt  # accepted fallback
|
|-- data/
|   |-- raw/
|   |   |-- low/<latitude>_<longitude>/
|   |   |-- moderate/<latitude>_<longitude>/
|   |   `-- high/<latitude>_<longitude>/
|   |
|   `-- processed/
|       |-- low/<latitude>_<longitude>/
|       |   |-- frame_000__YYYY-MM-DD.png
|       |   |-- frame_metadata.csv
|       |   |-- processing_metadata.json
|       |   |-- gee_features.csv
|       |   `-- gee_targets.csv
|       |-- moderate/<latitude>_<longitude>/
|       `-- high/<latitude>_<longitude>/
|
|-- notebooks/
|   |-- 01_video_to_cropped_frames.ipynb
|   |-- 02_extract_gee_features.ipynb
|   |-- 03_feature_engineering.ipynb
|   |-- 04_model_training.ipynb
|   |-- 05_model_evaluation.ipynb
|   `-- 06_demo_new_region.ipynb
|
|-- src/
|   |-- preprocessing/process_raw_videos.py
|   |-- features/
|   |-- models/
|   |-- evaluation/
|   `-- demo/
|
|-- models/
|-- outputs/
|-- report/
|-- slides/
|-- requirements.txt
`-- README.md
```

---

## 6. Metadata Design

Each processed sample stores metadata inside its own processed sample folder.

### 6.1 Frame Metadata

```text
data/processed/<label>/<latitude>_<longitude>/frame_metadata.csv
```

Columns:

```text
frame_index
timestamp_seconds
acquisition_date
image_file
width_px
height_px
crop_percent
```

Example:

```text
0,0,2016-06-08,frame_000__2016-06-08.png,922,922,5.0
```

### 6.2 Processing Metadata

```text
data/processed/<label>/<latitude>_<longitude>/processing_metadata.json
```

Fields:

```text
latitude
longitude
label
sample_id
raw_dir
raw_video
raw_timeline
processed_dir
frame_count
crop_percent
status
processed_at
gee_features_csv
```

`gee_features.csv` and `gee_targets.csv` are created later by `02_extract_gee_features.ipynb` and saved into the same processed sample folder.
---

## 7. Feature Engineering

### 7.1 Image Time-Series Feature

A time-series image model will be used to learn visual changes across time. The main purpose is to estimate whether surrounding urban/city growth is moving toward the center agricultural region.

Possible image-sequence models:

```text
CNN + LSTM
CNN + GRU
ConvLSTM
CNN feature extractor + Gaussian Process regression/classification
```

Output from this model:

```text
urban_growth_probability
urban_growth_confidence_bound
visual_land_change_score
```

These values will become additional features for the final sustainable agriculture classifier.

### 7.2 Geospatial and Environmental Features

Additional numerical features will be extracted based on the center coordinate.

Feature groups:

```text
Vegetation:
NDVI, EVI, NDWI, moisture stress

Climate:
rainfall, temperature, humidity, dry days, heavy-rain days

Terrain:
elevation, slope, lowland/flood-prone indicator

SAR:
VV, VH, VV/VH ratio, waterlogging signal

Accessibility:
distance to road, nearby population, nearby settlement, nearby city growth

Risk:
flood chance, nearby lake/river/water body, extreme rainfall condition
```

### 7.3 Final Feature Table

The final dataset will combine:

```text
image_time_series_features
+ GEE environmental features
+ elevation features
+ rainfall/weather features
+ SAR/flood features
+ accessibility/context features
```

The final model input becomes:

```text
X = [urban_growth_probability, NDVI, NDWI, rainfall, temperature, elevation, slope, flood risk, road distance, population context, ...]
y = sustainable_agriculture_class
```

---

## 8. Model Training

The final classifier will compare several machine learning models:

```text
Logistic Regression
Random Forest
Support Vector Machine
XGBoost / LightGBM if time allows
```

The output class is:

```text
Low / Moderate / High
```

Because the dataset size is small, classical ML models are more realistic than a large deep-learning model. The deep learning part should be used mainly for extracting visual time-series features, while the final decision model should use tabular ML methods.

---

## 9. Evaluation

The models will be evaluated using:

```text
accuracy
macro F1-score
precision
recall
confusion matrix
cross-validation if dataset size allows
```

Since the classes are ordered from Low to High, the project may also report:

```text
mean absolute class error
```

Example:

```text
Actual: High
Predicted: Moderate
Error distance: 1 class
```

The best model will be selected based on:

```text
highest macro F1-score
lowest class-distance error
most reasonable feature importance
best generalization on unseen testing regions
```

---

## 10. Demonstration

For the final demo, a new region that is not used in training or testing will be selected.

Demo input:

```text
Sentinel-2 agriculture-layer time-series images
center latitude and longitude
extracted environmental/geospatial features
```

Demo process:

```text
1. Place MP4 and timeline TXT in raw_to_be_processed/.
2. Run 01_video_to_cropped_frames.ipynb to archive raw files and create 5%-cropped PNG frames.
3. Run 02_extract_gee_features.ipynb to save gee_features.csv and gee_targets.csv in the same processed sample folder.
4. Build image time-series and tabular feature inputs.
5. Predict urban/city growth possibility.
6. Combine visual and environmental features.
7. Predict sustainable agriculture class.
8. Display final result and explanation.
```

Demo output example:

```text
Region ID: candidate_001
Predicted class: High sustainable agriculture suitability
Urban growth risk: Medium
Flood risk: Low
Vegetation condition: Good
Terrain suitability: Moderate
Final recommendation: Potentially suitable, but field validation is required.
```

---

## 11. Key Project Limitation

This project does not prove that a region can definitely remain productive for durian or any plantation crop. It estimates sustainable agriculture suitability based on satellite image similarity, environmental features, terrain condition, rainfall, and surrounding development patterns.

Since the image data is downloaded as MP4 and converted into screenshots, the image model uses lossy visual data. Therefore, numerical environmental features from Google Earth Engine and other APIs are necessary to improve reliability.

---

## 12. Expected Contribution

The project contributes a practical decision-support pipeline for Malaysian agriculture by combining:

```text
Sentinel-2 satellite image time series
environmental feature extraction
urban-growth forecasting
terrain and flood-risk indicators
machine learning classification
```

The system can help estimate whether a land region has low, moderate, or high sustainable agriculture suitability before detailed field investigation.



---

## 13. Leakage-Safe GEE Feature Outputs

`02_extract_gee_features.ipynb` must physically separate model inputs from future targets:

```text
data/processed/<label>/<latitude>_<longitude>/gee_observations.csv
data/processed/<label>/<latitude>_<longitude>/gee_features.csv   # model X only
data/processed/<label>/<latitude>_<longitude>/gee_targets.csv    # future/t+1 targets only
data/processed/<label>/<latitude>_<longitude>/gee_feature_metadata.json
```

The future/t+1 targets must stay in gee_targets.csv and must never be written into gee_features.csv.
