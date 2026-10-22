"""Calibrated ensemble of gradient boosting and time-decayed Dixon-Coles."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from src.config import DEFAULT_MODEL_PATH, MODEL_CONFIG
from src.dixon_coles import TimeDecayedDixonColes, walk_forward_dc_probs
from src.evaluate import TARGET_MAP, evaluate_predictions
from src.features import DC_COLS, FEATURE_COLS, MARKET_COLS, FeatureBuilder
from src.market import compute_kelly_edge, odds_are_valid
from src.names import normalize_team_name

logger = logging.getLogger(__name__)

BLEND_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


def make_classifier(max_iter: int = 200) -> CalibratedClassifierCV:
    """Gradient boosting (handles NaN features natively) wrapped in sigmoid calibration."""
    base = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=42,
    )
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)


def blend(ml_probs: np.ndarray, dc_probs: np.ndarray, ml_weight: float) -> np.ndarray:
    """Linear pool of ML and Dixon-Coles probabilities; rows without a DC estimate use ML alone."""
    ml_probs = np.atleast_2d(ml_probs)
    dc_probs = np.atleast_2d(dc_probs)
    has_dc = np.isfinite(dc_probs).all(axis=1, keepdims=True)
    mixed = np.where(has_dc, ml_weight * ml_probs + (1.0 - ml_weight) * np.nan_to_num(dc_probs), ml_probs)
    return mixed / mixed.sum(axis=1, keepdims=True)


class MatchPredictorEngine:
    """End-to-end match predictor: walk-forward Dixon-Coles + Elo/form/market features + calibrated GBM.

    Pickled artifacts execute code on load; only load models you trained yourself.
    """

    def __init__(self, n_splits: int = MODEL_CONFIG["cv_splits"]):
        self.n_splits = n_splits
        self.dc_model = TimeDecayedDixonColes()
        self.feature_builder = FeatureBuilder()
        self.ml_model = make_classifier()
        self.ml_weight_: float = 0.55
        self.feature_cols_: list[str] = list(FEATURE_COLS)
        self.cv_metrics_: dict[str, Any] = {}
        self.n_matches_: int = 0
        self.last_match_date_: str | None = None
        self.is_fitted_: bool = False

    @property
    def teams(self) -> list[str]:
        return list(self.dc_model.teams_)

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        """Walk-forward CV to pick the blend weight and measure skill, then a final fit on all data."""
        data = df.copy()
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date", kind="stable").reset_index(drop=True)

        logger.info("Computing walk-forward Dixon-Coles probabilities for %d matches...", len(data))
        dc_probs = walk_forward_dc_probs(data)

        logger.info("Engineering chronological Elo, rolling form, and de-vigged market features...")
        feat_df = self.feature_builder.transform_historical(data, dc_probs=dc_probs)

        # Train only once Dixon-Coles has warmed up (cold-start rows carry no DC signal), and drop
        # feature columns that are entirely missing, e.g. a CSV without bookmaker odds.
        warm = feat_df[DC_COLS].notna().all(axis=1).to_numpy()
        if warm.sum() < 60:
            raise ValueError("Not enough history for walk-forward validation; supply more matches")
        feat_df = feat_df[warm].reset_index(drop=True)
        self.feature_cols_ = [c for c in FEATURE_COLS if feat_df[c].notna().any()]

        X = feat_df[self.feature_cols_].to_numpy(dtype=float)
        y = feat_df["FTR"].map(TARGET_MAP).to_numpy()
        dc_all = feat_df[DC_COLS].to_numpy(dtype=float)
        mkt_all = feat_df[MARKET_COLS].to_numpy(dtype=float)

        oof_ml = np.full((len(y), 3), np.nan)
        for train_idx, val_idx in TimeSeriesSplit(n_splits=self.n_splits).split(X):
            fold_model = make_classifier()
            fold_model.fit(X[train_idx], y[train_idx])
            oof_ml[val_idx] = fold_model.predict_proba(X[val_idx])
        scored = np.isfinite(oof_ml).all(axis=1)

        # Pick the ML/Dixon-Coles mix that minimises out-of-fold log loss.
        losses = {
            w: log_loss(y[scored], blend(oof_ml[scored], dc_all[scored], w), labels=[0, 1, 2]) for w in BLEND_GRID
        }
        self.ml_weight_ = float(min(losses, key=losses.get))

        # Compare every model on the same matches so the numbers are like-for-like.
        common = scored & np.isfinite(mkt_all).all(axis=1)
        mask = common if common.sum() >= 30 else scored
        # Naive baseline: H/D/A frequencies from matches before the first scored one (Laplace-smoothed).
        base_rates = np.bincount(y[: int(np.argmax(mask))], minlength=3) + 1.0
        self.cv_metrics_ = {
            "ml_weight": self.ml_weight_,
            "ensemble": evaluate_predictions(y[mask], blend(oof_ml[mask], dc_all[mask], self.ml_weight_)),
            "gradient_boosting_only": evaluate_predictions(y[mask], oof_ml[mask]),
            "dixon_coles_baseline": evaluate_predictions(y[mask], dc_all[mask]),
            "base_rate_baseline": evaluate_predictions(
                y[mask], np.tile(base_rates / base_rates.sum(), (int(mask.sum()), 1))
            ),
        }
        if mask is common:
            self.cv_metrics_["bookmaker_market"] = evaluate_predictions(y[mask], mkt_all[mask])

        logger.info("Fitting final models on %d matches...", len(data))
        self.dc_model = TimeDecayedDixonColes().fit(data)
        self.ml_model = make_classifier()
        self.ml_model.fit(X, y)
        self.n_matches_ = len(data)
        self.last_match_date_ = data["Date"].max().strftime("%Y-%m-%d")
        self.is_fitted_ = True
        logger.info(
            "Ensemble CV metrics (ML weight %.2f): %s", self.ml_weight_, json.dumps(self.cv_metrics_["ensemble"])
        )
        return self.cv_metrics_

    def predict_fixture(
        self,
        home_team: str,
        away_team: str,
        odds: tuple[float, float, float] | None = None,
        match_date: str | pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        """Probability breakdown, likely scorelines, xG, and value edges for a fixture."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Run fit() or load() first.")

        h = normalize_team_name(home_team)
        a = normalize_team_name(away_team)
        unknown = [t for t in (h, a) if not self.dc_model.knows(t)]
        if unknown:
            raise ValueError(f"Unknown team(s): {unknown}. Known teams: {', '.join(self.teams)}")
        if h == a:
            raise ValueError("Home and away team must be different")
        if odds is not None and not odds_are_valid(*odds):
            raise ValueError(f"Odds must be three decimal prices > 1.0, got {odds}")

        X_fix = self.feature_builder.build_single_fixture_features(
            h, a, self.dc_model, odds=odds, match_date=match_date
        )
        ml_probs = self.ml_model.predict_proba(X_fix[self.feature_cols_].to_numpy(dtype=float))
        dc = self.dc_model.predict_1x2(h, a)
        p_h, p_d, p_a = blend(ml_probs, np.array([[dc["H"], dc["D"], dc["A"]]]), self.ml_weight_)[0]

        lam, mu = self.dc_model.expected_goals(h, a)
        score_mat = self.dc_model.predict_score_matrix(h, a)

        top_scorelines = []
        for idx in np.argsort(score_mat.ravel())[::-1][:5]:
            hg_s, ag_s = divmod(int(idx), score_mat.shape[1])
            top_scorelines.append({"score": f"{hg_s}-{ag_s}", "probability": round(float(score_mat[hg_s, ag_s]), 4)})

        rows_idx, cols_idx = np.indices(score_mat.shape)
        over_2_5 = float(np.sum(score_mat[(rows_idx + cols_idx) > 2]))
        btts_yes = float(np.sum(score_mat[(rows_idx > 0) & (cols_idx > 0)]))

        value_analysis = {}
        if odds is not None:
            value_analysis = {
                "H": compute_kelly_edge(float(p_h), odds[0]),
                "D": compute_kelly_edge(float(p_d), odds[1]),
                "A": compute_kelly_edge(float(p_a), odds[2]),
            }

        return {
            "home_team": h,
            "away_team": a,
            "probabilities": {
                "home_win": round(float(p_h), 4),
                "draw": round(float(p_d), 4),
                "away_win": round(float(p_a), 4),
            },
            "component_probabilities": {
                "gradient_boosting": [round(float(p), 4) for p in ml_probs[0]],
                "dixon_coles": [round(dc[k], 4) for k in ("H", "D", "A")],
                "ml_weight": self.ml_weight_,
            },
            "expected_goals": {"home_xg": round(lam, 2), "away_xg": round(mu, 2)},
            "elo_ratings": {
                "home_elo": round(self.feature_builder.elo_.get(h, 1500.0), 1),
                "away_elo": round(self.feature_builder.elo_.get(a, 1500.0), 1),
            },
            "markets": {"over_2_5_prob": round(over_2_5, 4), "btts_prob": round(btts_yes, 4)},
            "top_scorelines": top_scorelines,
            "value_bets": value_analysis,
        }

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved trained model artifact to %s", path)

    @staticmethod
    def load(path: Path = DEFAULT_MODEL_PATH) -> MatchPredictorEngine:
        """Loads a pickled engine. Never load artifacts from untrusted sources."""
        with open(path, "rb") as f:
            engine = pickle.load(f)
        if not isinstance(engine, MatchPredictorEngine):
            raise TypeError(f"{path} does not contain a MatchPredictorEngine")
        return engine
