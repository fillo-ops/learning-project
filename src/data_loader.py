"""Loading, validation and cleaning of historical match CSVs (Football-Data.co.uk format)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.names import normalize_team_name

logger = logging.getLogger(__name__)

OPTIONAL_NUMERIC_COLS = (
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HCC",
    "ACC",
    "HPoss",
    "APoss",
    "B365H",
    "B365D",
    "B365A",
)


def parse_match_dates(dates: pd.Series) -> pd.Series:
    """Parses match dates, treating ISO (YYYY-MM-DD) as year-first and everything else as day-first.

    Football-Data.co.uk uses DD/MM/YYYY, but CSVs written by this project use ISO.
    Parsing ISO strings with dayfirst=True silently swaps day and month
    (2023-08-11 -> 8 Nov), so the two formats are handled separately.
    """
    as_str = dates.astype(str).str.strip()
    is_iso = as_str.str.match(r"^\d{4}-\d{2}-\d{2}")
    parsed = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    if is_iso.any():
        parsed[is_iso] = pd.to_datetime(as_str[is_iso].str[:10], format="%Y-%m-%d", errors="coerce")
    if (~is_iso).any():
        parsed[~is_iso] = pd.to_datetime(as_str[~is_iso], format="mixed", dayfirst=True, errors="coerce")
    return parsed


class DataLoader:
    """Safely loads, validates, and cleans historical football CSV datasets."""

    REQUIRED_COLS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")

    def load_csv_safely(self, file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find dataset at {path}")
        df = pd.read_csv(path, encoding_errors="replace")
        logger.info("Loaded %d rows from %s", len(df), path.name)
        return self.clean_match_data(df)

    def load_many(self, paths: list[str | Path]) -> pd.DataFrame:
        """Loads and concatenates several season CSVs, de-duplicating overlapping fixtures."""
        frames = [self.load_csv_safely(p) for p in paths]
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
        return combined.sort_values("Date", kind="stable").reset_index(drop=True)

    def clean_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        cleaned = df.dropna(subset=list(self.REQUIRED_COLS)).copy()
        cleaned["Date"] = parse_match_dates(cleaned["Date"])
        bad_dates = cleaned["Date"].isna()
        if bad_dates.any():
            logger.warning("Dropping %d rows with unparseable dates", int(bad_dates.sum()))
            cleaned = cleaned[~bad_dates]

        cleaned["HomeTeam"] = cleaned["HomeTeam"].map(normalize_team_name)
        cleaned["AwayTeam"] = cleaned["AwayTeam"].map(normalize_team_name)
        cleaned["FTHG"] = pd.to_numeric(cleaned["FTHG"], errors="coerce")
        cleaned["FTAG"] = pd.to_numeric(cleaned["FTAG"], errors="coerce")
        cleaned = cleaned.dropna(subset=["FTHG", "FTAG"])
        cleaned["FTHG"] = cleaned["FTHG"].astype(int)
        cleaned["FTAG"] = cleaned["FTAG"].astype(int)
        cleaned["FTR"] = cleaned["FTR"].astype(str).str.strip().str.upper()
        cleaned = cleaned[cleaned["FTR"].isin(["H", "D", "A"])]

        # The result column must agree with the score; a mismatch means corrupt data.
        implied = cleaned["FTHG"].sub(cleaned["FTAG"]).map(lambda gd: "H" if gd > 0 else ("D" if gd == 0 else "A"))
        mismatched = implied != cleaned["FTR"]
        if mismatched.any():
            logger.warning("Dropping %d rows where FTR disagrees with the score", int(mismatched.sum()))
            cleaned = cleaned[~mismatched]

        for col in OPTIONAL_NUMERIC_COLS:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        return cleaned.sort_values("Date", kind="stable").reset_index(drop=True)
