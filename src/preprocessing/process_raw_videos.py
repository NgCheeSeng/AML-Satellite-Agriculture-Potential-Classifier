"""Convert raw Copernicus MP4 samples into labelled cropped frame datasets.

Expected incoming files:

    raw_to_be_processed/
      3.5165528_101.9364861_high.mp4
      3.5165528_101.9364861.txt

The MP4 filename carries the label. The preferred TXT name omits the label, but
the legacy/current `<latitude>_<longitude>_<label>.txt` style is also accepted.
The TXT file contains one acquisition date per line. Line 1 maps to timestamp
0s, line 2 maps to 1s, and so on.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable


LABELS = {"low", "moderate", "high"}
SAMPLE_INDEX_FIELDS = [
    "sample_id",
    "label",
    "latitude",
    "longitude",
    "processed_dir",
    "frame_count",
    "start_date",
    "end_date",
    "frame_metadata_csv",
    "processing_metadata_json",
    "gee_observations_csv",
    "gee_features_csv",
    "gee_targets_csv",
    "gee_feature_metadata_json",
]
VIDEO_RE = re.compile(
    r"^(?P<latitude>-?\d+(?:\.\d+)?)_(?P<longitude>-?\d+(?:\.\d+)?)_"
    r"(?P<label>low|moderate|high)\.mp4$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RawSample:
    latitude: str
    longitude: str
    label: str
    video_path: Path
    timeline_path: Path

    @property
    def coordinate_id(self) -> str:
        return f"{self.latitude}_{self.longitude}"

    @property
    def sample_id(self) -> str:
        return build_sample_id(self.latitude, self.longitude, self.label)


def build_sample_id(latitude: str | float, longitude: str | float, label: str) -> str:
    return f"{latitude}_{longitude}_{label.lower()}"


def parse_video_name(video_path: Path) -> tuple[str, str, str]:
    match = VIDEO_RE.match(video_path.name)
    if not match:
        raise ValueError(
            f"Invalid video filename: {video_path.name}. Expected "
            "<latitude>_<longitude>_<low|moderate|high>.mp4"
        )
    latitude = match.group("latitude")
    longitude = match.group("longitude")
    label = match.group("label").lower()
    return latitude, longitude, label


def read_timeline(timeline_path: Path) -> list[str]:
    if not timeline_path.exists():
        raise FileNotFoundError(f"Missing timeline file: {timeline_path}")

    dates = [
        line.strip()
        for line in timeline_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not dates:
        raise ValueError(f"Timeline file is empty: {timeline_path}")

    for value in dates:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid date {value!r} in {timeline_path}. Expected YYYY-MM-DD."
            ) from exc
    return dates


def discover_samples(inbox_dir: Path) -> list[RawSample]:
    if not inbox_dir.exists():
        raise FileNotFoundError(f"Incoming raw folder does not exist: {inbox_dir}")

    samples: list[RawSample] = []
    for video_path in sorted(inbox_dir.glob("*.mp4")):
        latitude, longitude, label = parse_video_name(video_path)
        timeline_path = inbox_dir / f"{latitude}_{longitude}.txt"
        if not timeline_path.exists():
            legacy_timeline_path = inbox_dir / f"{latitude}_{longitude}_{label}.txt"
            if legacy_timeline_path.exists():
                timeline_path = legacy_timeline_path
        samples.append(
            RawSample(
                latitude=latitude,
                longitude=longitude,
                label=label,
                video_path=video_path,
                timeline_path=timeline_path,
            )
        )
    return samples


def archive_raw_files(sample: RawSample, data_dir: Path) -> tuple[Path, Path, Path]:
    raw_dir = data_dir / "raw" / sample.label / sample.coordinate_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    archived_video = raw_dir / sample.video_path.name
    archived_timeline = raw_dir / f"{sample.coordinate_id}.txt"
    shutil.copy2(sample.video_path, archived_video)
    shutil.copy2(sample.timeline_path, archived_timeline)
    return raw_dir, archived_video, archived_timeline


def crop_border(frame, crop_percent: float):
    if crop_percent < 0 or crop_percent >= 50:
        raise ValueError("crop_percent must be >= 0 and < 50")

    height, width = frame.shape[:2]
    x_margin = int(round(width * crop_percent / 100.0))
    y_margin = int(round(height * crop_percent / 100.0))
    if x_margin == 0 and y_margin == 0:
        return frame
    return frame[y_margin : height - y_margin, x_margin : width - x_margin]


def expected_frame_path(processed_dir: Path, index: int, acquisition_date: str) -> Path:
    return processed_dir / f"frame_{index:03d}__{acquisition_date}.png"


def is_processed(
    processed_dir: Path,
    dates: Iterable[str],
    crop_percent: float,
) -> bool:
    metadata_path = processed_dir / "processing_metadata.json"
    if not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if metadata.get("status") != "processed":
        return False
    if float(metadata.get("crop_percent", -1)) != float(crop_percent):
        return False

    expected = [
        expected_frame_path(processed_dir, index, acquisition_date)
        for index, acquisition_date in enumerate(dates)
    ]
    return all(path.exists() for path in expected)


def extract_cropped_frames(
    sample: RawSample,
    data_dir: Path,
    crop_percent: float = 5.0,
    force: bool = False,
) -> dict:
    dates = read_timeline(sample.timeline_path)
    processed_dir = data_dir / "processed" / sample.label / sample.coordinate_id

    if not force and is_processed(processed_dir, dates, crop_percent):
        return {
            "sample_id": sample.sample_id,
            "coordinate_id": sample.coordinate_id,
            "label": sample.label,
            "status": "skipped",
            "reason": "already processed",
            "processed_dir": str(processed_dir),
            "frame_count": len(dates),
        }

    raw_dir, archived_video, archived_timeline = archive_raw_files(sample, data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for MP4 frame extraction. "
            "Install it with: pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(str(archived_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {archived_video}")

    frame_rows: list[dict[str, str | int | float]] = []
    try:
        for index, acquisition_date in enumerate(dates):
            timestamp_seconds = index
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not read frame {index} at {timestamp_seconds}s "
                    f"from {archived_video}"
                )

            cropped = crop_border(frame, crop_percent)
            output_path = expected_frame_path(processed_dir, index, acquisition_date)
            ok = cv2.imwrite(str(output_path), cropped)
            if not ok:
                raise RuntimeError(f"Could not write frame: {output_path}")

            height, width = cropped.shape[:2]
            frame_rows.append(
                {
                    "frame_index": index,
                    "timestamp_seconds": timestamp_seconds,
                    "acquisition_date": acquisition_date,
                    "image_file": output_path.name,
                    "width_px": width,
                    "height_px": height,
                    "crop_percent": crop_percent,
                }
            )
    finally:
        cap.release()

    frame_metadata = processed_dir / "frame_metadata.csv"
    with frame_metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "timestamp_seconds",
                "acquisition_date",
                "image_file",
                "width_px",
                "height_px",
                "crop_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_rows)

    metadata = {
        "latitude": float(sample.latitude),
        "longitude": float(sample.longitude),
        "label": sample.label,
        "sample_id": sample.sample_id,
        "coordinate_id": sample.coordinate_id,
        "raw_dir": str(raw_dir),
        "raw_video": str(archived_video),
        "raw_timeline": str(archived_timeline),
        "processed_dir": str(processed_dir),
        "frame_count": len(frame_rows),
        "crop_percent": crop_percent,
        "status": "processed",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "gee_observations_csv": str(processed_dir / "gee_observations.csv"),
        "gee_features_csv": str(processed_dir / "gee_features.csv"),
        "gee_targets_csv": str(processed_dir / "gee_targets.csv"),
        "gee_feature_metadata_json": str(processed_dir / "gee_feature_metadata.json"),
    }
    (processed_dir / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _load_processing_metadata(metadata_path: Path) -> dict:
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata file: {metadata_path}") from exc


def _read_frame_dates(frame_metadata_path: Path) -> tuple[int, str, str]:
    if not frame_metadata_path.exists():
        return 0, "", ""
    with frame_metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dates = [row.get("acquisition_date", "") for row in rows if row.get("acquisition_date")]
    if not dates:
        return len(rows), "", ""
    return len(rows), min(dates), max(dates)


def build_sample_index(data_dir: str | Path = "data") -> list[dict[str, str | int | float]]:
    data_path = Path(data_dir)
    processed_root = data_path / "processed"
    rows: list[dict[str, str | int | float]] = []

    for label in sorted(LABELS):
        label_dir = processed_root / label
        if not label_dir.exists():
            continue
        for sample_dir in sorted(path for path in label_dir.iterdir() if path.is_dir()):
            metadata_path = sample_dir / "processing_metadata.json"
            if not metadata_path.exists():
                continue
            metadata = _load_processing_metadata(metadata_path)
            coordinate_id = str(metadata.get("coordinate_id") or sample_dir.name)
            latitude = metadata.get("latitude")
            longitude = metadata.get("longitude")
            if latitude is None or longitude is None:
                try:
                    latitude_text, longitude_text = coordinate_id.split("_", 1)
                    latitude = float(latitude_text)
                    longitude = float(longitude_text)
                except ValueError:
                    latitude = ""
                    longitude = ""
            frame_metadata_path = sample_dir / "frame_metadata.csv"
            frame_count, start_date, end_date = _read_frame_dates(frame_metadata_path)
            if not frame_count:
                frame_count = int(metadata.get("frame_count", 0) or 0)
            rows.append(
                {
                    "sample_id": build_sample_id(latitude, longitude, label),
                    "label": label,
                    "latitude": latitude,
                    "longitude": longitude,
                    "processed_dir": str(sample_dir),
                    "frame_count": frame_count,
                    "start_date": start_date,
                    "end_date": end_date,
                    "frame_metadata_csv": str(frame_metadata_path),
                    "processing_metadata_json": str(metadata_path),
                    "gee_observations_csv": str(sample_dir / "gee_observations.csv"),
                    "gee_features_csv": str(sample_dir / "gee_features.csv"),
                    "gee_targets_csv": str(sample_dir / "gee_targets.csv"),
                    "gee_feature_metadata_json": str(sample_dir / "gee_feature_metadata.json"),
                }
            )
    return rows


def write_sample_index(data_dir: str | Path = "data") -> Path:
    data_path = Path(data_dir)
    processed_root = data_path / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    index_path = processed_root / "sample_index.csv"
    rows = build_sample_index(data_path)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def process_inbox(
    inbox_dir: str | Path = "raw_to_be_processed",
    data_dir: str | Path = "data",
    crop_percent: float = 5.0,
    force: bool = False,
) -> list[dict]:
    inbox_path = Path(inbox_dir)
    data_path = Path(data_dir)
    samples = discover_samples(inbox_path)
    if not samples:
        raise FileNotFoundError(f"No MP4 files found in {inbox_path}")

    results = []
    for sample in samples:
        results.append(
            extract_cropped_frames(
                sample=sample,
                data_dir=data_path,
                crop_percent=crop_percent,
                force=force,
            )
        )
    write_sample_index(data_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw Copernicus MP4 files into cropped PNG frames."
    )
    parser.add_argument("--inbox", default="raw_to_be_processed")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--crop-percent", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--write-sample-index-only",
        action="store_true",
        help="Rebuild data/processed/sample_index.csv without processing inbox videos.",
    )
    args = parser.parse_args()

    if args.write_sample_index_only:
        index_path = write_sample_index(args.data_dir)
        print(json.dumps({"sample_index_csv": str(index_path)}, indent=2))
        return

    results = process_inbox(
        inbox_dir=args.inbox,
        data_dir=args.data_dir,
        crop_percent=args.crop_percent,
        force=args.force,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
