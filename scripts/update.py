"""Download the latest Football-Data.co.uk season, validate it, merge it in and (optionally) retrain.

    python -m scripts.update --league E0 --season 2526             # download, validate, merge, retrain
    python -m scripts.update --league E0 --season 2526 --no-train  # validate + merge only
    python -m scripts.update --file E0.csv --check-only            # validate a local file

Exits with status 1 when validation finds errors and writes the report to
reports/data_validation.md (the weekly GitHub Action turns that into an issue).
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DEFAULT_CSV_PATH, REPORTS_DIR, ensure_dirs
from src.data_loader import DataLoader
from src.db import DatabaseManager
from src.validation import validate_matches

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def download(league: str, season: str, timeout: int = 30) -> pd.DataFrame:
    url = BASE_URL.format(season=season, league=league)
    req = urllib.request.Request(url, headers={"User-Agent": "football-predictor/2.3 (data update)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(raw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--file", type=Path, help="Validate / merge a local CSV instead of downloading")
    parser.add_argument("--league", default="E0", help="Football-Data league code (E0 = Premier League)")
    parser.add_argument("--season", default="2526", help="Season code, e.g. 2526 for 2025/26")
    parser.add_argument("--no-train", action="store_true", help="Do not retrain after merging")
    parser.add_argument("--check-only", action="store_true", help="Validate without merging or training")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_dirs()

    try:
        raw = pd.read_csv(args.file) if args.file else download(args.league, args.season)
    except (OSError, ValueError, pd.errors.ParserError) as exc:  # network / HTTP / parse failures
        report_text = f"Could not fetch data: {exc}"
        (REPORTS_DIR / "data_validation.md").write_text(f"## Data validation failed\n\n{report_text}\n")
        print(report_text, file=sys.stderr)
        return 1

    report = validate_matches(raw)
    source = args.file or BASE_URL.format(season=args.season, league=args.league)
    (REPORTS_DIR / "data_validation.md").write_text(f"## Data validation: {source}\n\n{report.to_markdown()}\n")
    print(report.to_markdown())
    if not report.ok:
        return 1
    if args.check_only:
        return 0

    new = DataLoader().clean_match_data(raw)
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_dir / f"{args.league}_{args.season}.csv", index=False)
    if DEFAULT_CSV_PATH.exists():
        merged = pd.concat([DataLoader().load_csv_safely(DEFAULT_CSV_PATH), new], ignore_index=True)
    else:
        merged = new
    merged = merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last").sort_values("Date")
    merged.to_csv(DEFAULT_CSV_PATH, index=False, date_format="%Y-%m-%d")
    DatabaseManager().save_matches(new)
    print(f"Merged {len(new)} matches; dataset now has {len(merged)}")

    if not args.no_train:
        from src.pipeline import run_full_pipeline

        run_full_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
