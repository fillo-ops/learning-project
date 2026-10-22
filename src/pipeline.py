"""Training pipeline and feature ablation study."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.analytics import TeamAnalytics
from src.config import (
    DEFAULT_CSV_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PLAYERS_PATH,
    MODEL_CONFIG,
    REPORTS_DIR,
    ensure_dirs,
)
from src.data_loader import DataLoader
from src.db import DatabaseManager
from src.dixon_coles import walk_forward_dc_probs
from src.evaluate import TARGET_MAP, evaluate_predictions
from src.features import DC_COLS, ELO_COLS, FEATURE_COLS, FORM_COLS, FeatureBuilder
from src.model import MatchPredictorEngine, make_classifier
from src.synthetic import generate_synthetic_matches, generate_synthetic_players

logger = logging.getLogger(__name__)


def load_or_create_dataset(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Loads the match CSV, generating a synthetic one first if none exists."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.info("No CSV found at %s. Generating 3 synthetic seasons...", csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_matches(n_seasons=3).to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    return DataLoader().load_csv_safely(csv_path)


def run_full_pipeline(csv_path: Path = DEFAULT_CSV_PATH, model_path: Path = DEFAULT_MODEL_PATH) -> MatchPredictorEngine:
    """Loads data, persists it to SQLite, trains and evaluates the engine, and saves artifacts."""
    ensure_dirs()
    df = load_or_create_dataset(csv_path)
    DatabaseManager().save_matches(df)

    engine = MatchPredictorEngine()
    metrics = engine.fit(df)
    engine.save(model_path)

    report_path = REPORTS_DIR / "cv_evaluation.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved evaluation report to %s", report_path)
    return engine


def load_or_train(model_path: Path = DEFAULT_MODEL_PATH) -> MatchPredictorEngine:
    if Path(model_path).exists():
        return MatchPredictorEngine.load(model_path)
    return run_full_pipeline(model_path=model_path)


def load_analytics(
    csv_path: Path = DEFAULT_CSV_PATH, players_path: Path = DEFAULT_PLAYERS_PATH, **simulator_kwargs
) -> TeamAnalytics:
    """Builds TeamAnalytics from the match CSV and the player CSV.

    When no player CSV exists, a synthetic squad table is generated so the player
    features can be demonstrated; pass real player data for real analysis.
    """
    matches = load_or_create_dataset(csv_path)
    if Path(players_path).exists():
        players = pd.read_csv(players_path)
    else:
        logger.warning("No player CSV at %s; using synthetic players for the demo", players_path)
        players = generate_synthetic_players(matches)
    return TeamAnalytics(matches, players, **simulator_kwargs)


def run_ablation_study(df: pd.DataFrame, n_splits: int = MODEL_CONFIG["cv_splits"]) -> pd.DataFrame:
    """Walk-forward metrics across progressively richer feature sets (all leak-free)."""
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date", kind="stable").reset_index(drop=True)
    feat_df = FeatureBuilder().transform_historical(data, dc_probs=walk_forward_dc_probs(data))
    # Same warm-up rule as the engine: start once Dixon-Coles has enough history.
    feat_df = feat_df[feat_df[DC_COLS].notna().all(axis=1)].reset_index(drop=True)
    y = feat_df["FTR"].map(TARGET_MAP).to_numpy()

    feature_groups = {
        "1. Rolling form + rest days": FORM_COLS,
        "2. + Elo ratings": ELO_COLS + FORM_COLS,
        "3. + Walk-forward Dixon-Coles": ELO_COLS + FORM_COLS + DC_COLS,
        "4. + De-vigged market (full)": FEATURE_COLS,
    }

    splits = list(TimeSeriesSplit(n_splits=n_splits).split(feat_df))
    results = []
    for group_name, cols in feature_groups.items():
        cols = [c for c in cols if feat_df[c].notna().any()]
        X = feat_df[cols].to_numpy(dtype=float)
        probs_list, y_list = [], []
        for train_idx, val_idx in splits:
            clf = make_classifier(max_iter=150)
            clf.fit(X[train_idx], y[train_idx])
            probs_list.append(clf.predict_proba(X[val_idx]))
            y_list.append(y[val_idx])
        results.append(
            {"Feature Set": group_name, **evaluate_predictions(np.concatenate(y_list), np.vstack(probs_list))}
        )

    return pd.DataFrame(results)
