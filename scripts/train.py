"""Train the ensemble end to end: python -m scripts.train [--csv path/to/matches.csv]"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import DEFAULT_CSV_PATH
from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Match CSV (Football-Data.co.uk format)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    engine = run_full_pipeline(csv_path=args.csv)
    print(json.dumps(engine.cv_metrics_, indent=2))
