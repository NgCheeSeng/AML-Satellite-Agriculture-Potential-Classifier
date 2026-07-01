"""Run one live coordinate month+1 risk prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo_inference import DemoCoordinateConfig, predict_demo_coordinate


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Predict month+1 environmental proxy risk for one coordinate.")
    parser.add_argument("--latitude", type=float, required=True, help="Demo coordinate latitude.")
    parser.add_argument("--longitude", type=float, required=True, help="Demo coordinate longitude.")
    parser.add_argument("--sample-id", default="demo_coordinate", help="Output sample id for the demo coordinate.")
    parser.add_argument("--source-month", default=None, help="Optional source month as YYYY-MM-01. Defaults to latest complete month.")
    parser.add_argument("--history-months", type=int, default=None, help="Monthly history rows to extract. Default comes from project_config.yaml.")
    parser.add_argument("--allow-incomplete-source-month", action="store_true", help="Allow current/incomplete month as source_month.")
    parser.add_argument("--raw-csv", default=None, help="Optional existing raw monthly CSV for offline smoke testing.")
    parser.add_argument("--output-csv", default="reports/demo_coordinate_prediction.csv", help="Prediction CSV output path.")
    parser.add_argument("--config-path", default="config/project_config.yaml", help="Project config path.")
    return parser.parse_args()


def main() -> None:
    """Run the demo coordinate prediction."""

    args = parse_args()
    prediction = predict_demo_coordinate(DemoCoordinateConfig(
        latitude=args.latitude,
        longitude=args.longitude,
        sample_id=args.sample_id,
        source_month=args.source_month,
        history_months=args.history_months,
        allow_incomplete_source_month=args.allow_incomplete_source_month,
        output_csv=args.output_csv,
        config_path=args.config_path,
        raw_csv=args.raw_csv,
    ))
    print(prediction.to_string(index=False))


if __name__ == "__main__":
    main()
