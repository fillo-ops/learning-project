"""Monte Carlo matchup simulator: attack vs defence, with an explicit midfield battle.

Each simulated match draws, for both sides,

    shots      ~ Poisson(expected_shots * performance_noise)      (midfield / territory)
    on target  ~ Binomial(shots, p_on_target)                      (chance quality vs defence)
    goals      ~ Binomial(on_target, p_convert)                    (finishing vs goalkeeping)

using opponent-adjusted rates from TeamUnitRatings, optionally modified by player
absences (PlayerRatings) and a manual midfield shift for "what if" analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import SIMULATION_CONFIG
from src.names import normalize_team_name
from src.players import NEUTRAL_EFFECTS, PlayerRatings
from src.team_ratings import TeamUnitRatings

MAX_GOALS_BUCKET = 6


class MatchSimulator:
    """Simulates fixtures many times to show how attacks and defences fare against each other."""

    def __init__(
        self,
        ratings: TeamUnitRatings,
        players: PlayerRatings | None = None,
        n_sims: int = SIMULATION_CONFIG["n_sims"],
        performance_sd: float = SIMULATION_CONFIG["performance_sd"],
        seed: int = 2024,
    ):
        self.ratings = ratings
        self.players = players
        self.n_sims = int(n_sims)
        self.performance_sd = float(performance_sd)
        self.seed = seed

    # ------------------------------------------------------------------ rates
    def _effects(self, team: str, absent: list[str] | None) -> dict[str, float]:
        if not absent:
            return dict(NEUTRAL_EFFECTS)
        if self.players is None:
            raise ValueError("Player absences need player data (PlayerRatings)")
        return self.players.absence_effects(team, absent)

    def side_rates(
        self,
        home: str,
        away: str,
        midfield_shift: float = 0.0,
        home_absent: list[str] | None = None,
        away_absent: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Per-side expected shots, on-target share and conversion after all adjustments.

        `midfield_shift` is a log-scale tilt of the territorial battle toward the home side
        (0.1 is roughly +10% home shots and -10% away shots).
        """
        base = self.ratings.matchup_rates(home, away)
        eff = {"home": self._effects(home, home_absent), "away": self._effects(away, away_absent)}
        out = {}
        for side, opp, sign in (("home", "away", 1.0), ("away", "home", -1.0)):
            r, me, them = base[side], eff[side], eff[opp]
            shots = r["shots"] * me["shots_for"] * them["shots_against"] * np.exp(sign * midfield_shift)
            p_on = float(np.clip(r["p_on_target"] * them["on_target_against"], 0.02, 0.95))
            p_conv = float(np.clip(r["p_convert"] * me["finishing"] * them["conversion_against"], 0.01, 0.95))
            out[side] = {
                "shots": float(shots),
                "p_on_target": p_on,
                "p_convert": p_conv,
                "xg": float(shots * p_on * p_conv),
            }
        return out

    # ------------------------------------------------------------- simulation
    def _draw(self, rates: dict[str, dict[str, float]], rng: np.random.Generator) -> dict[str, np.ndarray]:
        sd = self.performance_sd
        draws = {}
        for side in ("home", "away"):
            r = rates[side]
            noise = np.exp(rng.normal(-0.5 * sd**2, sd, self.n_sims))  # mean-one log-normal
            shots = rng.poisson(r["shots"] * noise)
            sot = rng.binomial(shots, r["p_on_target"])
            goals = rng.binomial(sot, r["p_convert"])
            draws[side] = np.stack([shots, sot, goals])
        return draws

    def simulate(
        self,
        home: str,
        away: str,
        midfield_shift: float = 0.0,
        home_absent: list[str] | None = None,
        away_absent: list[str] | None = None,
    ) -> dict[str, Any]:
        """Runs `n_sims` matches and summarises attack, defence, midfield and scorer outcomes."""
        h, a = normalize_team_name(home), normalize_team_name(away)
        if h == a:
            raise ValueError("Home and away team must be different")
        rates = self.side_rates(h, a, midfield_shift, home_absent, away_absent)
        draws = self._draw(rates, np.random.default_rng(self.seed))
        hg, ag = draws["home"][2], draws["away"][2]

        teams = {"home": (h, a, True, home_absent), "away": (a, h, False, away_absent)}
        sides: dict[str, Any] = {}
        for side, (team, opp, at_home, absent) in teams.items():
            shots, sot, goals = draws[side]
            conceded = draws["away" if side == "home" else "home"][2]
            sides[side] = {
                "team": team,
                "attack": self._goal_summary(goals, shots, sot),
                "defence": self._defence_summary(team, opp, at_home, conceded),
                "goal_breakdown": self.ratings.goal_decomposition(team, opp, at_home).to_dict(orient="index"),
                "midfield_effect": self._midfield_effect(team, opp, at_home, rates[side]["xg"]),
                "expected_rates": {k: round(v, 3) for k, v in rates[side].items()},
            }
            if self.players is not None:
                try:
                    sides[side]["players"] = {
                        "unit_ratings": self.players.team_units(team, absent),
                        "absence_effects": self.players.absence_effects(team, absent),
                        "top_scorers": self._scorer_odds(team, absent, goals),
                    }
                except ValueError:
                    pass  # no player data for this team

        scores = pd.Series(list(zip(hg.tolist(), ag.tolist()))).value_counts(normalize=True).head(5)
        return {
            "home_team": h,
            "away_team": a,
            "n_sims": self.n_sims,
            "midfield_shift": midfield_shift,
            "outcome": {
                "home_win": round(float(np.mean(hg > ag)), 4),
                "draw": round(float(np.mean(hg == ag)), 4),
                "away_win": round(float(np.mean(hg < ag)), 4),
            },
            "top_scorelines": [
                {"score": f"{x}-{y}", "probability": round(float(p), 4)} for (x, y), p in scores.items()
            ],
            "unit_comparison": self.unit_comparison(h, a),
            "home": sides["home"],
            "away": sides["away"],
        }

    @staticmethod
    def _goal_summary(goals: np.ndarray, shots: np.ndarray, sot: np.ndarray) -> dict[str, Any]:
        dist = np.bincount(np.minimum(goals, MAX_GOALS_BUCKET), minlength=MAX_GOALS_BUCKET + 1) / len(goals)
        labels = [str(k) for k in range(MAX_GOALS_BUCKET)] + [f"{MAX_GOALS_BUCKET}+"]
        return {
            "mean_goals": round(float(goals.mean()), 3),
            "sd_goals": round(float(goals.std()), 3),
            "goal_distribution": {lab: round(float(p), 4) for lab, p in zip(labels, dist)},
            "p_score_1_plus": round(float(np.mean(goals >= 1)), 4),
            "p_score_2_plus": round(float(np.mean(goals >= 2)), 4),
            "p_score_3_plus": round(float(np.mean(goals >= 3)), 4),
            "goals_90th_percentile": int(np.percentile(goals, 90)),
            "max_goals_seen": int(goals.max()),
            "mean_shots": round(float(shots.mean()), 2),
            "mean_shots_on_target": round(float(sot.mean()), 2),
        }

    def _defence_summary(self, team: str, opp: str, at_home: bool, conceded: np.ndarray) -> dict[str, Any]:
        goals = self.ratings.layers_["goals"]
        # What this defence concedes against a league-average attack at the same venue
        typical = float(np.exp(goals.intercept + goals.allow[team] + (0.0 if at_home else goals.home)))
        expected = float(conceded.mean())
        return {
            "expected_conceded": round(expected, 3),
            "typical_conceded_vs_average_attack": round(typical, 3),
            "clean_sheet_prob": round(float(np.mean(conceded == 0)), 4),
            "p_concede_3_plus": round(float(np.mean(conceded >= 3)), 4),
            "verdict": (
                "under more pressure than usual" if expected > 1.1 * typical
                else "better protected than usual" if expected < 0.9 * typical
                else "about its usual workload"
            ),
        }  # fmt: skip

    def _midfield_effect(self, team: str, opp: str, at_home: bool, xg: float) -> dict[str, float]:
        """Goals gained or lost purely through the shot-volume (midfield) battle."""
        s = self.ratings.layers_["shots"]
        volume_edge = s.gen[team] + s.allow[opp]
        neutral_xg = xg / np.exp(volume_edge)
        return {
            "shot_volume_multiplier": round(float(np.exp(volume_edge)), 3),
            "xg_with_neutral_midfield": round(float(neutral_xg), 3),
            "goals_from_midfield_edge": round(float(xg - neutral_xg), 3),
        }

    def _scorer_odds(self, team: str, absent: list[str] | None, goals: np.ndarray, top: int = 5) -> list[dict]:
        w = self.players.scorer_weights(team, absent)  # type: ignore[union-attr]
        counts = np.bincount(goals)
        k = np.arange(len(counts))
        pk = counts / counts.sum()
        rows = []
        for player, share in w.sort_values(ascending=False).head(top).items():
            p_score = 1.0 - float(np.sum(pk * (1.0 - share) ** k))
            rows.append(
                {
                    "player": player,
                    "share_of_team_goals": round(float(share), 3),
                    "expected_goals": round(float(share * goals.mean()), 3),
                    "p_anytime_scorer": round(p_score, 4),
                }
            )
        return rows

    # ----------------------------------------------------- matchup analytics
    def unit_comparison(self, home: str, away: str) -> list[dict[str, Any]]:
        """Head-to-head unit battles (indices: 100 = league average)."""
        ph, pa = self.ratings.unit_profile(home), self.ratings.unit_profile(away)
        h, a = normalize_team_name(home), normalize_team_name(away)
        battles = [
            ("Midfield control", h, "midfield", a, "midfield"),
            (f"{h} attack vs {a} defence", h, "attack", a, "defence"),
            (f"{a} attack vs {h} defence", a, "attack", h, "defence"),
            (f"{h} chance quality vs {a} block", h, "chance_quality", a, "shot_quality_allowed"),
            (f"{a} chance quality vs {h} block", a, "chance_quality", h, "shot_quality_allowed"),
            (f"{h} finishing vs {a} goalkeeping", h, "finishing", a, "goalkeeping"),
            (f"{a} finishing vs {h} goalkeeping", a, "finishing", h, "goalkeeping"),
        ]
        prof = {h: ph, a: pa}
        rows = []
        for label, t1, u1, t2, u2 in battles:
            v1, v2 = prof[t1][u1], prof[t2][u2]
            rows.append(
                {"battle": label, t1: v1, t2: v2, "edge": t1 if v1 > v2 else t2, "margin": round(abs(v1 - v2), 1)}
            )
        return rows

    def midfield_sensitivity(
        self, home: str, away: str, shifts: tuple[float, ...] = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
    ) -> pd.DataFrame:
        """How expected goals and results move as the midfield battle tilts (same random numbers per row)."""
        rows = []
        for shift in shifts:
            rates = self.side_rates(home, away, midfield_shift=shift)
            draws = self._draw(rates, np.random.default_rng(self.seed))
            hg, ag = draws["home"][2], draws["away"][2]
            rows.append(
                {
                    "midfield_shift": shift,
                    "home_shots": round(rates["home"]["shots"], 1),
                    "away_shots": round(rates["away"]["shots"], 1),
                    "home_goals": round(float(hg.mean()), 3),
                    "away_goals": round(float(ag.mean()), 3),
                    "home_win": round(float(np.mean(hg > ag)), 4),
                    "draw": round(float(np.mean(hg == ag)), 4),
                    "away_win": round(float(np.mean(hg < ag)), 4),
                }
            )
        return pd.DataFrame(rows)

    def attack_vs_all_defences(self, team: str, at_home: bool = True) -> pd.DataFrame:
        """Expected goals and scoring probabilities for `team`'s attack against every defence."""
        t = normalize_team_name(team)
        rows = []
        for opp in self.ratings.teams_:
            if opp == t:
                continue
            home, away = (t, opp) if at_home else (opp, t)
            xg = self.side_rates(home, away)["home" if at_home else "away"]["xg"]
            rows.append(
                {
                    "opponent": opp,
                    "opponent_defence": self.ratings.unit_profile(opp)["defence"],
                    "expected_goals": round(xg, 3),
                    "p_score_1_plus": round(float(1 - poisson.cdf(0, xg)), 4),
                    "p_score_2_plus": round(float(1 - poisson.cdf(1, xg)), 4),
                    "p_score_3_plus": round(float(1 - poisson.cdf(2, xg)), 4),
                }
            )
        return pd.DataFrame(rows).sort_values("expected_goals", ascending=False).reset_index(drop=True)
