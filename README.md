# Satellite Agriculture Potential Classifier

This repository contains the source code for an AML final project that builds a satellite image time-series pipeline for classifying Malaysian agricultural land potential into three labels:

```text
low / moderate / high
```

The source code is hosted on GitHub:

```text
git@github.com:NgCheeSeng/AML-Satellite-Agriculture-Potential-Classifier.git
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
    gee_features.csv

notebooks/
  01_video_to_cropped_frames.ipynb

src/
  preprocessing/process_raw_videos.py

scripts/
  upload_dataset_to_hf.py
```

The `data/` and `raw_to_be_processed/` folders are intentionally ignored by Git. Dataset files should be shared through Hugging Face, not committed to GitHub.

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

## Upload Dataset to Hugging Face

Authenticate once with a Hugging Face token that has write permission for the dataset repo:

```powershell
conda run -n aml hf auth login
```

Preview files before upload:

```powershell
conda run -n aml python scripts/upload_dataset_to_hf.py --dry-run
```

Upload `data/` to the dataset repository:

```powershell
conda run -n aml python scripts/upload_dataset_to_hf.py
```

The script defaults to:

```text
Aki298/AML-Satellite-Imagery-Malaysia-Copernicus
```

You can also use `HF_TOKEN` instead of `hf auth login`:

```powershell
$env:HF_TOKEN="your_token_here"
conda run -n aml python scripts/upload_dataset_to_hf.py
```

For very large future uploads, use the Hugging Face large-folder uploader:

```powershell
conda run -n aml hf upload-large-folder Aki298/AML-Satellite-Imagery-Malaysia-Copernicus --repo-type=dataset data --num-workers=8
```

## Notes

- Labels are `low`, `moderate`, and `high`.
- Current frame extraction crops 5% from each border to remove browser overlay/watermark areas.
- `gee_features.csv` is created later by the Google Earth Engine feature extraction notebook and should be saved inside the same processed sample folder.
