"""Predict a fixture from the command line.

python -m scripts.predict --home Arsenal --away Chelsea --odds 2.05 3.50 3.60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.config import DEFAULT_MODEL_PATH
from src.db import DatabaseManager
from src.pipeline import load_or_train, run_full_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Football match outcome predictor")
    parser.add_argument("--home", default="Arsenal", help="Home team name")
    parser.add_argument("--away", default="Chelsea", help="Away team name")
    parser.add_argument("--odds", type=float, nargs=3, metavar=("H", "D", "A"), help="Decimal odds for value analysis")
    parser.add_argument("--date", help="Match date (YYYY-MM-DD), used for rest-day features")
    parser.add_argument("--retrain", action="store_true", help="Retrain even if a saved model exists")
    parser.add_argument("--log", action="store_true", help="Store the prediction in the SQLite predictions table")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    engine = run_full_pipeline() if args.retrain else load_or_train(DEFAULT_MODEL_PATH)

    try:
        prediction = engine.predict_fixture(
            args.home, args.away, odds=tuple(args.odds) if args.odds else None, match_date=args.date
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.log:
        DatabaseManager().save_prediction(prediction, match_date=args.date)
    print(json.dumps(prediction, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
