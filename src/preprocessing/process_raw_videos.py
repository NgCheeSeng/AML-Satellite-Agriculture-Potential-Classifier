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
    def sample_id(self) -> str:
        return f"{self.latitude}_{self.longitude}"


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
    raw_dir = data_dir / "raw" / sample.label / sample.sample_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    archived_video = raw_dir / sample.video_path.name
    archived_timeline = raw_dir / f"{sample.sample_id}.txt"
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
    processed_dir = data_dir / "processed" / sample.label / sample.sample_id

    if not force and is_processed(processed_dir, dates, crop_percent):
        return {
            "sample_id": sample.sample_id,
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
        "raw_dir": str(raw_dir),
        "raw_video": str(archived_video),
        "raw_timeline": str(archived_timeline),
        "processed_dir": str(processed_dir),
        "frame_count": len(frame_rows),
        "crop_percent": crop_percent,
        "status": "processed",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "gee_features_csv": str(processed_dir / "gee_features.csv"),
    }
    (processed_dir / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


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
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw Copernicus MP4 files into cropped PNG frames."
    )
    parser.add_argument("--inbox", default="raw_to_be_processed")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--crop-percent", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = process_inbox(
        inbox_dir=args.inbox,
        data_dir=args.data_dir,
        crop_percent=args.crop_percent,
        force=args.force,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
