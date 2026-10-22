"""Paths and hyperparameter configuration."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = DATA_DIR / "football.db"
DEFAULT_CSV_PATH = DATA_DIR / "matches.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "ensemble_predictor.pkl"

MODEL_CONFIG = {
    "dixon_coles_halflife_days": 120.0,
    "dixon_coles_max_goals": 10,
    "dixon_coles_l2": 0.005,
    # Walk-forward Dixon-Coles: refit cadence and minimum history before first fit
    "dixon_coles_refit_days": 28,
    "dixon_coles_min_matches": 100,
    "elo_k_factor": 20.0,
    "elo_home_advantage": 65.0,
    "rolling_window": 5,
    "cv_splits": 5,
    "kelly_fraction": 0.25,
    "min_ev_threshold": 0.02,
}

DEFAULT_PLAYERS_PATH = DATA_DIR / "players.csv"

# How player absences translate into team-unit changes. These are modelling assumptions,
# not fitted values: tune them if you have data to calibrate against.
PLAYER_CONFIG = {
    "min_minutes": 270,  # players below this are rated but flagged as small-sample
    "replacement_level": 0.6,  # a stand-in produces this fraction of the absent player's output
    "midfield_weight": 0.5,  # share of a midfielder's absence felt in shot volume for/against
    "defence_weight": 0.6,  # share of a defender's absence felt in opponent shots on target
    "keeper_weight": 0.15,  # extra goals per shot on target conceded with the second-choice keeper
}

SIMULATION_CONFIG = {
    "n_sims": 20000,
    "performance_sd": 0.12,  # match-to-match noise on each side's shot volume (log scale)
}


def ensure_dirs() -> None:
    """Creates runtime output directories (kept out of import time on purpose)."""
    for directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
