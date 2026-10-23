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

UNIT_CONFIG = {
    # Attack index = goals blended with shots on target; on-target counts are steadier than goals.
    "attack_sot_weight": 0.5,
    # With xG data the attack index becomes npxG / shots on target / non-penalty goals in these shares.
    "attack_weights_with_xg": {"npxg": 0.5, "sot": 0.25, "goals": 0.25},
    # Pre-appointment matches of a team that changed manager count this much in its ratings.
    "manager_change_discount": 0.4,
    # Elo ratings are pulled this far toward the mean between seasons (squad turnover).
    "season_regression": 0.2,
}

DEFAULT_PLAYERS_PATH = DATA_DIR / "players.csv"
# Optional: non-league fixtures (Date, Team, Competition) and manager appointments (Team, Date)
DEFAULT_FIXTURES_PATH = DATA_DIR / "fixtures.csv"
DEFAULT_MANAGERS_PATH = DATA_DIR / "managers.csv"

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
    "game_state": True,  # simulate six 15-minute blocks with scoreline and red-card effects
}

# Scoreline effects per 15-minute block: (chance-volume multiplier, chance-quality multiplier),
# keyed by the team's own goal margin. Leaders sit deeper (fewer, better counter chances);
# trailers push on (more, lower-quality chances). These are priors from the football-analytics
# literature, not fitted values: without minute-level data they cannot be estimated here.
# The simulator re-scales base rates so full-match expected goals still match the ratings.
GAME_STATE_CONFIG = {
    "n_blocks": 6,
    # Relative chance volume per block: goal rates climb through a match (tiring legs, chasing teams).
    # Normalised to mean 1, so this changes timing, not totals.
    "block_profile": (0.85, 0.95, 1.0, 0.98, 1.05, 1.17),
    "state_effects": {
        -2: (1.18, 0.90),  # trailing by 2+
        -1: (1.12, 0.94),
        0: (1.00, 1.00),
        1: (0.88, 1.08),
        2: (0.80, 1.12),  # leading by 2+
    },
    "red_card_own_volume": 0.70,  # a side down to 10 men creates 30% fewer chances...
    "red_card_opp_volume": 1.30,  # ...and its opponent 30% more
    "league_red_rate": 0.05,  # red cards per team per match when the data has no HR/AR columns
    "red_prior_matches": 15,  # shrink team / referee red-card rates toward the league mean
    "calibration_sims": 6000,
}

# Penalties are modelled separately from open play when the data has PK columns.
PENALTY_CONFIG = {"league_conversion": 0.78, "prior_matches": 20}


def ensure_dirs() -> None:
    """Creates runtime output directories (kept out of import time on purpose)."""
    for directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
