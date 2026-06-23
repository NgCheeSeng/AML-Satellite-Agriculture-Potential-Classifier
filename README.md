# Satellite Sustainable Agriculture Classifier

This repository contains the source code for an AML final project that builds a satellite image time-series pipeline for predicting sustainable Malaysian agriculture land into three labels:

```text
low / moderate / high
```

The source code is hosted on GitHub:

```text
git@github.com:NgCheeSeng/AML-Satellite-Sustainable-Agriculture-Classifier.git
```

The dataset is hosted separately on Hugging Face:

```text
https://huggingface.co/datasets/Aki298/AML-Satellite-Imagery-Malaysia-Copernicus
```

## Project Structure

```text
raw_to_be_processed/
  <latitude>_<longitude>_<label>.mp4
  <latitude>_<longitude>.txt
  <latitude>_<longitude>_<label>.txt  # accepted fallback

data/
  raw/<label>/<latitude>_<longitude>/
  processed/<label>/<latitude>_<longitude>/
    frame_000__YYYY-MM-DD.png
    frame_metadata.csv
    processing_metadata.json
    gee_observations.csv
    gee_features.csv        # model X only
    gee_targets.csv         # future/t+1 targets only
    gee_feature_metadata.json

notebooks/
  01_video_to_cropped_frames.ipynb
  02_extract_gee_features.ipynb

src/
  preprocessing/process_raw_videos.py
  features/gee_features.py
```

The `data/` and `raw_to_be_processed/` folders are intentionally ignored by Git. Dataset files should be pulled from Hugging Face, not committed to GitHub.

## Environment Setup

Use the project conda environment:

```powershell
conda activate aml
python -m pip install -r requirements.txt
```

Or run commands without activating:

```powershell
conda run -n aml python -m pip install -r requirements.txt
```

## Pull Dataset from Hugging Face

The project dataset is stored in:

```text
Aki298/AML-Satellite-Imagery-Malaysia-Copernicus
```

Download the dataset into the local `data/` folder:

```powershell
conda run -n aml python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Aki298/AML-Satellite-Imagery-Malaysia-Copernicus', repo_type='dataset', local_dir='data', allow_patterns=['raw/**','processed/**'])"
```

After pulling, the expected local dataset structure is:

```text
data/
  raw/
  processed/
```

If `data/` already exists, Hugging Face will update matching downloaded files, but local files that no longer exist in the remote dataset may remain. For a clean refresh, move or remove the old local `data/` folder before downloading again.

## Preprocess MP4 Samples

Place raw MP4 and TXT timeline files in `raw_to_be_processed/`.

Expected naming:

```text
3.5165528_101.9364861_high.mp4
3.5165528_101.9364861.txt
```

The fallback TXT name is also accepted:

```text
3.5165528_101.9364861_high.txt
```

The TXT file should contain one acquisition date per line. Line 1 maps to video timestamp `0s`, line 2 maps to `1s`, and so on.

Run the pipeline:

```powershell
conda run -n aml python -m src.preprocessing.process_raw_videos
```

The pipeline archives raw files into:

```text
data/raw/<label>/<latitude>_<longitude>/
```

It saves 5%-cropped PNG frames and metadata into:

```text
data/processed/<label>/<latitude>_<longitude>/
```

It also rebuilds the central sample index used by the GEE pipeline:

```text
data/processed/sample_index.csv
```


## Notes

- Labels are `low`, `moderate`, and `high`.
- Current frame extraction crops 5% from each border to remove browser overlay/watermark areas.
- `02_extract_gee_features.ipynb` creates `gee_observations.csv`, leakage-safe model inputs in `gee_features.csv`, future/t+1 targets in `gee_targets.csv`, and reproducibility details in `gee_feature_metadata.json`.


## Extract GEE Features

Set your Google Earth Engine project ID before launching Jupyter, then run notebook 02:

```powershell
$env:GEE_PROJECT_ID="your-gee-project-id"
jupyter notebook
```

If Jupyter is already running, set `GEE_PROJECT_ID` directly in the first config cell of `02_extract_gee_features.ipynb` and rerun the initialization cell.

`02_extract_gee_features.ipynb` writes per-sample files under `data/processed/<label>/<latitude>_<longitude>/`:

```text
gee_observations.csv
gee_features.csv
gee_targets.csv
gee_feature_metadata.json
```

`gee_features.csv` contains model input features only. `gee_targets.csv` contains future/t+1 target columns only, so future data cannot accidentally leak into training inputs.
