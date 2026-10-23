"""Generate a synthetic dataset: python -m scripts.make_synthetic [--seasons 3] [--seed 42]"""

from __future__ import annotations

import argparse

from src.config import DEFAULT_CSV_PATH, DEFAULT_PLAYERS_PATH, ensure_dirs
from src.db import DatabaseManager
from src.synthetic import generate_synthetic_matches, generate_synthetic_players

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--league-only", action="store_true", help="Skip the Champions League fixtures")
    args = parser.parse_args()
    ensure_dirs()
    df = generate_synthetic_matches(
        n_seasons=args.seasons, seed=args.seed, include_champions_league=not args.league_only
    )
    df.to_csv(DEFAULT_CSV_PATH, index=False, date_format="%Y-%m-%d")
    DatabaseManager().save_matches(df)
    players = generate_synthetic_players(df)
    players.to_csv(DEFAULT_PLAYERS_PATH, index=False)
    print(f"Generated {len(df)} matches at {DEFAULT_CSV_PATH}")
    print(f"Generated {len(players)} player rows at {DEFAULT_PLAYERS_PATH}")
