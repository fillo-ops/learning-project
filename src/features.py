"""Leak-free chronological Elo, rolling form, congestion, league-table stakes and market features."""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from src.config import MODEL_CONFIG, UNIT_CONFIG
from src.context import SEASON_GAP_DAYS, FixtureCalendar, LeagueTable
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
# Fixture congestion (all competitions when a calendar is supplied) and league-table motivation
CONTEXT_COLS = ["games14_diff", "h_dead_rubber", "a_dead_rubber", "table_pts_diff", "stake_gap_diff"]
FEATURE_COLS = ELO_COLS + FORM_COLS + CONTEXT_COLS + MARKET_COLS + DC_COLS

DEFAULT_REST_DAYS = 7
NEUTRAL_STAKES = {
    "games_left": 0.0,
    "pts_to_top4": 0.0,
    "drop_margin": 0.0,
    "pts_to_leader": 0.0,
    "dead_rubber": 0.0,
    "stake_gap": 0.0,
    "points": 0.0,
}


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
        season_regression: float = UNIT_CONFIG["season_regression"],
        calendar: FixtureCalendar | None = None,
        manager_changes: dict[str, list[pd.Timestamp]] | None = None,
    ):
        self.k_factor = k_factor
        self.home_elo_adv = home_elo_adv
        self.rolling_window = rolling_window
        self.season_regression = season_regression
        self.calendar = calendar or FixtureCalendar()
        self.manager_changes = manager_changes or {}
        self.elo_: dict[str, float] = {}
        self.last_played_: dict[str, pd.Timestamp] = {}
        self.league_dates_: dict[str, list[pd.Timestamp]] = {}
        self.team_history_: dict[str, list[dict[str, float]]] = {}
        self.table_ = LeagueTable()
        self.last_date_: pd.Timestamp | None = None

    def _reset(self) -> None:
        self.elo_.clear()
        self.last_played_.clear()
        self.league_dates_.clear()
        self.team_history_.clear()
        self.table_.reset(20)
        self.last_date_ = None

    def _start_new_season(self) -> None:
        """Summer reset: regress Elo toward the mean (squad turnover) and start a fresh table."""
        r = self.season_regression
        self.elo_ = {t: 1500.0 + (1.0 - r) * (e - 1500.0) for t, e in self.elo_.items()}
        self.table_.reset(len(self.table_.points) or None)

    def _apply_manager_changes(self, team: str, date: pd.Timestamp) -> None:
        """Clears rolling form when a new manager took over since the team's last match."""
        last = self.last_played_.get(team)
        if last is None:
            return
        if any(last < d <= date for d in self.manager_changes.get(team, [])):
            self.team_history_[team] = []

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
        """Days since the team's previous game in any competition (league + calendar)."""
        if date is None:
            return DEFAULT_REST_DAYS
        candidates = [d for d in (self.last_played_.get(team), self.calendar.last_before(team, date)) if d is not None]
        if not candidates:
            return DEFAULT_REST_DAYS
        return int(np.clip((date - max(candidates)).days, 2, 14))

    def _games_last_14(self, team: str, date: pd.Timestamp | None) -> int:
        if date is None:
            return 2
        start = date - pd.Timedelta(days=14)
        league = sum(1 for d in self.league_dates_.get(team, [])[-6:] if start <= d < date)
        return league + self.calendar.games_between(team, start, date)

    def _pre_match_row(
        self,
        h: str,
        a: str,
        date: pd.Timestamp | None,
        mkt: tuple[float, float, float],
        dc: tuple[float, float, float],
        league_match: bool = True,
    ) -> dict[str, float]:
        h_elo, a_elo, elo_diff, elo_exp_h = self._elo_state(h, a)
        h_gf, h_ga, h_sot, h_pts = self._get_rolling_stats(h)
        a_gf, a_ga, a_sot, a_pts = self._get_rolling_stats(a)
        # League-table stakes only apply to league matches (a Champions League tie has its own stakes)
        if league_match:
            hs, as_ = self.table_.stakes(h), self.table_.stakes(a)
        else:
            hs = as_ = NEUTRAL_STAKES
        h_g14, a_g14 = self._games_last_14(h, date), self._games_last_14(a, date)
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
            "h_games14": float(h_g14),
            "a_games14": float(a_g14),
            "games14_diff": float(h_g14 - a_g14),
            "h_dead_rubber": hs["dead_rubber"],
            "a_dead_rubber": as_["dead_rubber"],
            "h_pts_to_top4": hs["pts_to_top4"],
            "a_pts_to_top4": as_["pts_to_top4"],
            "table_pts_diff": hs["points"] - as_["points"],
            "h_stake_gap": hs["stake_gap"],
            "a_stake_gap": as_["stake_gap"],
            "stake_gap_diff": hs["stake_gap"] - as_["stake_gap"],
            "h_games_left": hs["games_left"],
            "mkt_prob_h": mkt[0],
            "mkt_prob_d": mkt[1],
            "mkt_prob_a": mkt[2],
            "dc_prob_h": dc[0],
            "dc_prob_d": dc[1],
            "dc_prob_a": dc[2],
        }

    def _update_state(
        self,
        h: str,
        a: str,
        date: pd.Timestamp,
        hg: int,
        ag: int,
        h_sot: float,
        a_sot: float,
        league_match: bool = True,
    ) -> None:
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
        self.league_dates_.setdefault(h, []).append(date)
        self.league_dates_.setdefault(a, []).append(date)
        if league_match:
            self.table_.update(h, a, hg, ag)

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

        self._reset()
        # With a Competition column, the most common competition is the domestic league;
        # other competitions (e.g. the Champions League) feed Elo, form and congestion but not the table.
        domestic = (
            data["Competition"].mode().iloc[0] if "Competition" in data and data["Competition"].notna().any() else None
        )

        rows: list[dict[str, float]] = []
        for i, row in enumerate(data.to_dict(orient="records")):
            h = normalize_team_name(row["HomeTeam"])
            a = normalize_team_name(row["AwayTeam"])
            dt = row["Date"]
            if self.last_date_ is not None and (dt - self.last_date_).days > SEASON_GAP_DAYS:
                self._start_new_season()
            self.last_date_ = dt
            self._apply_manager_changes(h, dt)
            self._apply_manager_changes(a, dt)
            comp = row.get("Competition")
            league_match = domestic is None or pd.isna(comp) or comp == domestic
            mkt = remove_bookmaker_vig(row.get("B365H"), row.get("B365D"), row.get("B365A"))
            rows.append(self._pre_match_row(h, a, dt, mkt, tuple(dc_probs[i]), league_match))

            # Update state strictly AFTER recording pre-match features
            if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
                h_sot = _finite_or(row.get("HST"), max(hg, 4))
                a_sot = _finite_or(row.get("AST"), max(ag, 3))
                self._update_state(h, a, dt, hg, ag, h_sot, a_sot, league_match)

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
        state = self
        if date is not None and self.last_date_ is not None and (date - self.last_date_).days > SEASON_GAP_DAYS:
            state = copy.deepcopy(self)  # predicting into a new season: apply the summer reset to a copy
            state._start_new_season()
        if date is not None and (self.manager_changes.get(h) or self.manager_changes.get(a)):
            state = copy.deepcopy(state) if state is self else state
            state._apply_manager_changes(h, date)
            state._apply_manager_changes(a, date)
        # Table stakes only when both clubs are in the domestic league table
        in_league = not state.table_.points or (h in state.table_.points and a in state.table_.points)
        row = state._pre_match_row(h, a, date, mkt, (dc["H"], dc["D"], dc["A"]), in_league)
        return pd.DataFrame([row])[FEATURE_COLS]
