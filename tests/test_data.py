import numpy as np
import pandas as pd

from src.data_loader import DataLoader, parse_match_dates
from src.db import DatabaseManager
from src.synthetic import generate_synthetic_matches, round_robin_schedule


def test_iso_dates_are_not_day_month_swapped():
    parsed = parse_match_dates(pd.Series(["2023-08-11", "11/08/2023", "11/08/23"]))
    assert (parsed == pd.Timestamp("2023-08-11")).all()


def test_clean_drops_inconsistent_rows():
    raw = pd.DataFrame(
        {
            "Date": ["12/08/2023", "13/08/2023", "14/08/2023"],
            "HomeTeam": ["Man Utd", "Spurs", "Wolves"],
            "AwayTeam": ["Man City", "Arsenal", "Leeds"],
            "FTHG": [1, 2, 0],
            "FTAG": [1, 0, 1],
            "FTR": ["D", "A", "A"],  # second row contradicts its score
        }
    )
    clean = DataLoader().clean_match_data(raw)
    assert len(clean) == 2
    assert clean["HomeTeam"].tolist() == ["Manchester United", "Wolverhampton Wanderers"]


def test_round_robin_each_team_once_per_round():
    rounds = round_robin_schedule(20, np.random.default_rng(0))
    assert len(rounds) == 38
    for fixtures in rounds:
        teams = [t for pair in fixtures for t in pair]
        assert sorted(teams) == list(range(20))
    all_pairs = {pair for fixtures in rounds for pair in fixtures}
    assert len(all_pairs) == 20 * 19  # every ordered home/away pairing exactly once


def test_synthetic_csv_round_trip_preserves_order(tmp_path):
    df = generate_synthetic_matches(n_seasons=1)
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    loaded = DataLoader().load_csv_safely(path)
    assert loaded["Date"].min() == df["Date"].min()
    assert loaded["Date"].max() == df["Date"].max()


def test_db_round_trip_keeps_missing_odds_null(tmp_path):
    df = generate_synthetic_matches(n_seasons=1).head(5).copy()
    df.loc[0, "B365H"] = np.nan
    db = DatabaseManager(tmp_path / "t.db")
    assert db.save_matches(df) == 5
    assert db.save_matches(df) == 5  # upsert, not duplicate
    loaded = db.load_matches()
    assert len(loaded) == 5
    assert loaded["B365H"].isna().sum() == 1
