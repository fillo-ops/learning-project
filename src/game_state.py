"""In-play game-state simulation: scoreline effects and red cards over 15-minute blocks.

Each simulated match is split into blocks (six of 15 minutes by default). In every block a
side's chance volume and chance quality are scaled by its current goal margin (leaders sit
deeper, trailers push on) and by red cards. Red cards arrive at a per-block rate built from
the team's discipline record and, when known, the referee's strictness.

Base rates come from full-match averages, which already include all game states. Before
simulating, the per-side rates are re-scaled so the simulated full-match expected goals
match those averages; game state then redistributes *when* and *in which scorelines*
goals happen (e.g. more 1-0 / 1-1 results, fewer runaway scores) rather than how many.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import GAME_STATE_CONFIG
from src.names import normalize_team_name


class DisciplineModel:
    """Red-card rates per team and referee strictness, shrunk toward the league average."""

    def __init__(self, config: dict | None = None):
        self.config = {**GAME_STATE_CONFIG, **(config or {})}
        self.league_rate_: float = self.config["league_red_rate"]
        self.team_rate_: dict[str, float] = {}
        self.referee_factor_: dict[str, float] = {}
        self.has_data_: bool = False

    def fit(self, df: pd.DataFrame) -> DisciplineModel:
        if not {"HR", "AR"} <= set(df.columns):
            return self
        data = df.dropna(subset=["HR", "AR"])
        if data.empty:
            return self
        prior = self.config["red_prior_matches"]
        self.league_rate_ = float((data["HR"].sum() + data["AR"].sum()) / (2 * len(data)))
        rows = pd.concat(
            [
                pd.DataFrame({"team": data["HomeTeam"].map(normalize_team_name), "reds": data["HR"]}),
                pd.DataFrame({"team": data["AwayTeam"].map(normalize_team_name), "reds": data["AR"]}),
            ]
        )
        agg = rows.groupby("team")["reds"].agg(["sum", "count"])
        self.team_rate_ = ((agg["sum"] + prior * self.league_rate_) / (agg["count"] + prior)).to_dict()
        if "Referee" in data.columns:
            ref = data.assign(reds=data["HR"] + data["AR"]).groupby("Referee")["reds"].agg(["sum", "count"])
            per_match = (ref["sum"] + prior * 2 * self.league_rate_) / (ref["count"] + prior)
            self.referee_factor_ = (per_match / (2 * self.league_rate_)).to_dict()
        self.has_data_ = True
        return self

    def red_rate(self, team: str, referee: str | None = None) -> float:
        """Expected red cards per match for `team`, adjusted for the referee if known."""
        rate = self.team_rate_.get(normalize_team_name(team), self.league_rate_)
        if referee is not None:
            rate *= self.referee_factor_.get(referee, 1.0)
        return float(rate)


@dataclass
class BlockDraws:
    """Per-side simulation output (arrays over simulations, block arrays are sims x blocks)."""

    volume: np.ndarray
    sot: np.ndarray
    goals: np.ndarray
    goals_by_block: np.ndarray
    reds: np.ndarray
    penalty_goals_mean: float = 0.0


def _state_multipliers(margin: np.ndarray, effects: dict[int, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    m = np.clip(margin, -2, 2)
    vol = np.ones(margin.shape)
    qual = np.ones(margin.shape)
    for k, (v, q) in effects.items():
        hit = m == k
        vol[hit] = v
        qual[hit] = q
    return vol, qual


def simulate_blocks(
    rates: dict[str, dict[str, float]],
    red_rates: dict[str, float],
    n_sims: int,
    performance_sd: float,
    rng: np.random.Generator,
    config: dict | None = None,
    scales: dict[str, float] | None = None,
) -> dict[str, BlockDraws]:
    """Simulates matches block by block. `rates` holds per-side volume, p_on_target and p_convert."""
    cfg = {**GAME_STATE_CONFIG, **(config or {})}
    n_blocks = int(cfg["n_blocks"])
    effects = cfg["state_effects"]
    scales = scales or {"home": 1.0, "away": 1.0}
    sides = ("home", "away")
    noise = {s: np.exp(rng.normal(-0.5 * performance_sd**2, performance_sd, n_sims)) for s in sides}
    goals = {s: np.zeros(n_sims, dtype=int) for s in sides}
    volume = {s: np.zeros(n_sims, dtype=int) for s in sides}
    sot = {s: np.zeros(n_sims, dtype=int) for s in sides}
    by_block = {s: np.zeros((n_sims, n_blocks), dtype=int) for s in sides}
    reds = {s: np.zeros(n_sims, dtype=int) for s in sides}
    pen_goals = {s: np.zeros(n_sims, dtype=int) for s in sides}
    p_red_block = {s: 1.0 - np.exp(-red_rates[s] / n_blocks) for s in sides}
    profile = np.asarray(cfg.get("block_profile") or [1.0] * n_blocks, dtype=float)
    if len(profile) != n_blocks:
        raise ValueError("block_profile must have one entry per block")
    profile = profile / profile.mean()

    for b in range(n_blocks):
        new_goals = {}
        for s, o in (("home", "away"), ("away", "home")):
            r = rates[s]
            vol_mult, qual_mult = _state_multipliers(goals[s] - goals[o], effects)
            vol_mult = vol_mult * cfg["red_card_own_volume"] ** reds[s] * cfg["red_card_opp_volume"] ** reds[o]
            lam = r["volume"] * scales[s] * noise[s] * vol_mult * profile[b] / n_blocks
            v = rng.poisson(lam)
            on = rng.binomial(v, np.clip(r["p_on_target"] * qual_mult, 0.0, 0.98))
            g = rng.binomial(on, r["p_convert"])
            if r.get("pen_rate"):
                pg = rng.binomial(rng.poisson(r["pen_rate"] / n_blocks, n_sims), r["pen_conv"])
                pen_goals[s] += pg
                g = g + pg
            volume[s] += v
            sot[s] += on
            new_goals[s] = g
        for s in sides:
            goals[s] += new_goals[s]
            by_block[s][:, b] = new_goals[s]
            # A red card shown in this block affects the following blocks
            reds[s] += rng.random(n_sims) < p_red_block[s]
    return {s: BlockDraws(volume[s], sot[s], goals[s], by_block[s], reds[s], float(pen_goals[s].mean())) for s in sides}


def calibrated_scales(
    rates: dict[str, dict[str, float]],
    red_rates: dict[str, float],
    performance_sd: float,
    seed: int,
    config: dict | None = None,
    iterations: int = 5,
    n_sims: int | None = None,
) -> dict[str, float]:
    """Volume scales that make block-simulated mean goals equal each side's full-match expected goals.

    Uses the same seed (common random numbers) as the main run, so the fixed-point iteration
    converges on the actual simulated sample rather than chasing sampling noise.
    """
    cfg = {**GAME_STATE_CONFIG, **(config or {})}
    n = int(n_sims or cfg["calibration_sims"])
    scales = {"home": 1.0, "away": 1.0}
    target = {s: rates[s]["xg"] for s in scales}
    for _ in range(iterations):
        draws = simulate_blocks(rates, red_rates, n, performance_sd, np.random.default_rng(seed), cfg, scales)
        for s in scales:
            # Penalties are not scaled, so only the open-play part is calibrated.
            open_play = max(float(draws[s].goals.mean()) - draws[s].penalty_goals_mean, 1e-6)
            scales[s] *= target[s] / open_play
    return scales


def game_state_summary(draws: dict[str, BlockDraws], block_minutes: int = 15) -> dict[str, Any]:
    """How matches unfold: who scores first, time spent leading, comebacks, red cards, goal timing."""
    h, a = draws["home"], draws["away"]
    n_blocks = h.goals_by_block.shape[1]
    cum_h = np.cumsum(h.goals_by_block, axis=1)
    cum_a = np.cumsum(a.goals_by_block, axis=1)
    margin_before = np.concatenate([np.zeros((len(h.goals), 1), dtype=int), (cum_h - cum_a)[:, :-1]], axis=1)

    # First goal: the earliest block with a goal (same-block ties split by each side's goals in it)
    any_goal = (h.goals_by_block + a.goals_by_block) > 0
    first_block = np.where(any_goal.any(axis=1), any_goal.argmax(axis=1), -1)
    rows = np.arange(len(h.goals))
    hb = h.goals_by_block[rows, np.maximum(first_block, 0)]
    ab = a.goals_by_block[rows, np.maximum(first_block, 0)]
    home_first = (first_block >= 0) & (hb >= ab)
    away_first = (first_block >= 0) & (ab > hb)
    hw, aw = h.goals > a.goals, a.goals > h.goals

    def cond(p_event: np.ndarray, given: np.ndarray) -> float | None:
        return round(float(p_event[given].mean()), 4) if given.any() else None

    blocks = [f"{b * block_minutes}-{(b + 1) * block_minutes}" for b in range(n_blocks)]
    return {
        "p_goalless": round(float(np.mean(first_block < 0)), 4),
        "home_scores_first": round(float(home_first.mean()), 4),
        "away_scores_first": round(float(away_first.mean()), 4),
        "home_win_if_scores_first": cond(hw, home_first),
        "away_win_if_scores_first": cond(aw, away_first),
        "home_comeback_win": round(float(np.mean(hw & (margin_before.min(axis=1) < 0))), 4),
        "away_comeback_win": round(float(np.mean(aw & (margin_before.max(axis=1) > 0))), 4),
        "share_of_match_home_leading": round(float(np.mean(margin_before > 0)), 4),
        "share_of_match_level": round(float(np.mean(margin_before == 0)), 4),
        "share_of_match_away_leading": round(float(np.mean(margin_before < 0)), 4),
        "p_home_red_card": round(float(np.mean(h.reds > 0)), 4),
        "p_away_red_card": round(float(np.mean(a.reds > 0)), 4),
        "goals_by_period": {
            "home": dict(zip(blocks, np.round(h.goals_by_block.mean(axis=0), 3).tolist())),
            "away": dict(zip(blocks, np.round(a.goals_by_block.mean(axis=0), 3).tolist())),
        },
    }
