"""Leak-free chronological Elo, rolling form, rest-day and market features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import MODEL_CONFIG
from src.dixon_coles import TimeDecayedDixonColes, walk_forward_dc_probs
from src.market import remove_bookmaker_vig
from src.names import normalize_team_name

FORM_COLS = [
    "rest_diff",
    "h_roll_gf",
    "h_roll_ga",
    "h_roll_sot",
    "a_roll_gf",
    "a_roll_ga",
    "a_roll_sot",
    "form_pts_diff",
]
ELO_COLS = ["elo_diff", "elo_exp_h"]
DC_COLS = ["dc_prob_h", "dc_prob_d", "dc_prob_a"]
MARKET_COLS = ["mkt_prob_h", "mkt_prob_d", "mkt_prob_a"]
FEATURE_COLS = ELO_COLS + FORM_COLS + MARKET_COLS + DC_COLS

DEFAULT_REST_DAYS = 7


def elo_goal_multiplier(goal_diff: int) -> float:
    """World Football Elo margin multiplier: 1 for draws/one-goal wins, growing with the margin."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _finite_or(value: object, default: float) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


class FeatureBuilder:
    """Computes strictly pre-match features without future target leakage."""

    def __init__(
        self,
        k_factor: float = MODEL_CONFIG["elo_k_factor"],
        home_elo_adv: float = MODEL_CONFIG["elo_home_advantage"],
        rolling_window: int = MODEL_CONFIG["rolling_window"],
    ):
        self.k_factor = k_factor
        self.home_elo_adv = home_elo_adv
        self.rolling_window = rolling_window
        self.elo_: dict[str, float] = {}
        self.last_played_: dict[str, pd.Timestamp] = {}
        self.team_history_: dict[str, list[dict[str, float]]] = {}

    def _get_rolling_stats(self, team: str) -> tuple[float, float, float, float]:
        """Exponentially weighted (recent-heavy) goals for/against, shots on target and points."""
        hist = self.team_history_.get(team, [])[-self.rolling_window :]
        if not hist:
            return 1.35, 1.35, 4.2, 1.35
        weights = np.exp(np.linspace(-1.0, 0.0, len(hist)))
        weights /= weights.sum()
        stats = np.array([[m["gf"], m["ga"], m["sot"], m["pts"]] for m in hist], dtype=float)
        gf, ga, sot, pts = weights @ stats
        return float(gf), float(ga), float(sot), float(pts)

    def _elo_state(self, h: str, a: str) -> tuple[float, float, float, float]:
        h_elo = self.elo_.get(h, 1500.0)
        a_elo = self.elo_.get(a, 1500.0)
        elo_diff = (h_elo + self.home_elo_adv) - a_elo
        elo_exp_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        return h_elo, a_elo, elo_diff, elo_exp_h

    def _rest_days(self, team: str, date: pd.Timestamp | None) -> int:
        if date is None or team not in self.last_played_:
            return DEFAULT_REST_DAYS
        return int(np.clip((date - self.last_played_[team]).days, 2, 14))

    def _pre_match_row(
        self,
        h: str,
        a: str,
        date: pd.Timestamp | None,
        mkt: tuple[float, float, float],
        dc: tuple[float, float, float],
    ) -> dict[str, float]:
        h_elo, a_elo, elo_diff, elo_exp_h = self._elo_state(h, a)
        h_gf, h_ga, h_sot, h_pts = self._get_rolling_stats(h)
        a_gf, a_ga, a_sot, a_pts = self._get_rolling_stats(a)
        return {
            "h_elo": h_elo,
            "a_elo": a_elo,
            "elo_diff": elo_diff,
            "elo_exp_h": elo_exp_h,
            "rest_diff": float(self._rest_days(h, date) - self._rest_days(a, date)),
            "h_roll_gf": h_gf,
            "h_roll_ga": h_ga,
            "h_roll_sot": h_sot,
            "h_roll_pts": h_pts,
            "a_roll_gf": a_gf,
            "a_roll_ga": a_ga,
            "a_roll_sot": a_sot,
            "a_roll_pts": a_pts,
            "form_pts_diff": h_pts - a_pts,
            "mkt_prob_h": mkt[0],
            "mkt_prob_d": mkt[1],
            "mkt_prob_a": mkt[2],
            "dc_prob_h": dc[0],
            "dc_prob_d": dc[1],
            "dc_prob_a": dc[2],
        }

    def _update_state(self, h: str, a: str, date: pd.Timestamp, hg: int, ag: int, h_sot: float, a_sot: float) -> None:
        _, _, _, elo_exp_h = self._elo_state(h, a)
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = self.k_factor * elo_goal_multiplier(hg - ag) * (actual_h - elo_exp_h)
        self.elo_[h] = self.elo_.get(h, 1500.0) + delta
        self.elo_[a] = self.elo_.get(a, 1500.0) - delta

        h_pts = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
        a_pts = 3.0 if ag > hg else (1.0 if hg == ag else 0.0)
        self.team_history_.setdefault(h, []).append({"gf": hg, "ga": ag, "sot": h_sot, "pts": h_pts})
        self.team_history_.setdefault(a, []).append({"gf": ag, "ga": hg, "sot": a_sot, "pts": a_pts})
        self.last_played_[h] = date
        self.last_played_[a] = date

    def transform_historical(self, df: pd.DataFrame, dc_probs: np.ndarray | None = None) -> pd.DataFrame:
        """Builds pre-match features for every row, replaying history in date order.

        `dc_probs` must be out-of-sample Dixon-Coles probabilities aligned with
        `df` (see walk_forward_dc_probs); they are computed here when omitted.
        Fitting one Dixon-Coles model on all of `df` and using its probabilities
        as features would leak each match's own result into its features.
        """
        data = df.copy()
        data["Date"] = pd.to_datetime(data["Date"])
        if dc_probs is None:
            dc_probs = walk_forward_dc_probs(data)
        if len(dc_probs) != len(data):
            raise ValueError("dc_probs must have one row per match")
        order = np.argsort(data["Date"].to_numpy(), kind="stable")
        data = data.iloc[order].reset_index(drop=True)
        dc_probs = np.asarray(dc_probs, dtype=float)[order]

        self.elo_.clear()
        self.last_played_.clear()
        self.team_history_.clear()

        rows: list[dict[str, float]] = []
        for i, row in enumerate(data.to_dict(orient="records")):
            h = normalize_team_name(row["HomeTeam"])
            a = normalize_team_name(row["AwayTeam"])
            dt = row["Date"]
            mkt = remove_bookmaker_vig(row.get("B365H"), row.get("B365D"), row.get("B365A"))
            rows.append(self._pre_match_row(h, a, dt, mkt, tuple(dc_probs[i])))

            # Update state strictly AFTER recording pre-match features
            if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
                h_sot = _finite_or(row.get("HST"), max(hg, 4))
                a_sot = _finite_or(row.get("AST"), max(ag, 3))
                self._update_state(h, a, dt, hg, ag, h_sot, a_sot)

        feat_df = pd.DataFrame(rows, index=data.index)
        return pd.concat([data, feat_df], axis=1)

    def build_single_fixture_features(
        self,
        home_team: str,
        away_team: str,
        dc_model: TimeDecayedDixonColes,
        odds: tuple[float, float, float] | None = None,
        match_date: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Features for an upcoming fixture, using state after the last replayed match.

        Missing odds become NaN market features, exactly as in training, rather
        than being filled with model probabilities the classifier never saw.
        """
        h = normalize_team_name(home_team)
        a = normalize_team_name(away_team)
        dc = dc_model.predict_1x2(h, a)
        mkt = remove_bookmaker_vig(*odds) if odds is not None else (math.nan, math.nan, math.nan)
        date = pd.Timestamp(match_date) if match_date is not None else None
        row = self._pre_match_row(h, a, date, mkt, (dc["H"], dc["D"], dc["A"]))
        return pd.DataFrame([row])[FEATURE_COLS]
