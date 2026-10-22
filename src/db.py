"""SQLite persistence for teams, historical matches, and predictions."""

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DB_PATH

_MATCH_COLUMNS = {
    # db column -> dataframe column
    "match_date": "Date",
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "fthg": "FTHG",
    "ftag": "FTAG",
    "ftr": "FTR",
    "hs": "HS",
    "as_shots": "AS",
    "hst": "HST",
    "ast": "AST",
    "hcc": "HCC",
    "acc": "ACC",
    "hposs": "HPoss",
    "aposs": "APoss",
    "b365h": "B365H",
    "b365d": "B365D",
    "b365a": "B365A",
}


_MATCH_COLUMN_TYPES = {
    "match_date": "TEXT", "home_team": "TEXT", "away_team": "TEXT", "fthg": "INTEGER", "ftag": "INTEGER",
    "ftr": "TEXT", "hs": "INTEGER", "as_shots": "INTEGER", "hst": "INTEGER", "ast": "INTEGER", "hcc": "INTEGER",
    "acc": "INTEGER", "hposs": "REAL", "aposs": "REAL", "b365h": "REAL", "b365d": "REAL", "b365a": "REAL",
}  # fmt: skip


def _sql_value(value: Any) -> Any:
    """Converts pandas/numpy scalars to SQLite-friendly values; missing becomes NULL."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class DatabaseManager:
    """Manages SQLite persistence for teams, historical matches, and predictions."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS teams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                );

                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    fthg INTEGER,
                    ftag INTEGER,
                    ftr TEXT,
                    hs INTEGER,
                    as_shots INTEGER,
                    hst INTEGER,
                    ast INTEGER,
                    hcc INTEGER,
                    acc INTEGER,
                    hposs REAL,
                    aposs REAL,
                    b365h REAL,
                    b365d REAL,
                    b365a REAL,
                    UNIQUE(match_date, home_team, away_team)
                );

                CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    match_date TEXT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    prob_h REAL NOT NULL,
                    prob_d REAL NOT NULL,
                    prob_a REAL NOT NULL,
                    predicted_score TEXT,
                    actual_ftr TEXT
                );
                """
            )
            # Databases created before possession / chances were tracked get the new columns added.
            existing = {row[1] for row in conn.execute("PRAGMA table_info(matches)")}
            for col, sql_type in _MATCH_COLUMN_TYPES.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE matches ADD COLUMN {col} {sql_type}")

    def save_matches(self, df: pd.DataFrame) -> int:
        """Upserts match records. Missing optional stats/odds are stored as NULL, never invented."""
        records = df.copy()
        records["Date"] = pd.to_datetime(records["Date"]).dt.strftime("%Y-%m-%d")
        db_cols = list(_MATCH_COLUMNS)
        rows = [
            tuple(_sql_value(row.get(_MATCH_COLUMNS[c])) for c in db_cols) for row in records.to_dict(orient="records")
        ]
        teams = {(t,) for t in pd.concat([records["HomeTeam"], records["AwayTeam"]]).unique()}
        placeholders = ", ".join("?" for _ in db_cols)
        with closing(self._connect()) as conn, conn:
            conn.executemany("INSERT OR IGNORE INTO teams (name) VALUES (?)", sorted(teams))
            conn.executemany(
                f"INSERT OR REPLACE INTO matches ({', '.join(db_cols)}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)

    def load_matches(self) -> pd.DataFrame:
        """Loads all historical matches ordered chronologically."""
        select = ", ".join(f'{db} AS "{df}"' for db, df in _MATCH_COLUMNS.items())
        with closing(self._connect()) as conn:
            df = pd.read_sql_query(f"SELECT {select} FROM matches ORDER BY match_date ASC, id ASC", conn)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        return df

    def save_prediction(self, prediction: dict[str, Any], match_date: str | None = None) -> int:
        """Logs a prediction from MatchPredictorEngine.predict_fixture for later tracking."""
        probs = prediction["probabilities"]
        top = prediction.get("top_scorelines") or [{}]
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                """
                INSERT INTO predictions
                (match_date, home_team, away_team, prob_h, prob_d, prob_a, predicted_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_date,
                    prediction["home_team"],
                    prediction["away_team"],
                    probs["home_win"],
                    probs["draw"],
                    probs["away_win"],
                    top[0].get("score"),
                ),
            )
            return int(cur.lastrowid)
