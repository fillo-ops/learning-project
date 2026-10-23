"""Monte Carlo matchup simulator: attack vs defence, with an explicit midfield battle.

Each simulated match draws, for both sides,

    volume     ~ Poisson(expected_chances * performance_noise)    (midfield: chances created; shots if
                                                                  the data has no chances column)
    on target  ~ Binomial(volume, p_on_target)                     (chance quality vs defence)
    goals      ~ Binomial(on_target, p_convert)                    (finishing vs goalkeeping)

using opponent-adjusted rates from TeamUnitRatings, optionally modified by player
absences (PlayerRatings) and "what if" shifts in possession or the midfield battle.
Possession shifts are converted into chance volume using the possession elasticity
estimated from the match data.

By default matches are played in six 15-minute blocks (src/game_state.py) so scoreline
effects and red cards change each side's chance volume and quality as the match unfolds;
`game_state=False` reverts to a single full-match draw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import GAME_STATE_CONFIG, SIMULATION_CONFIG
from src.game_state import BlockDraws, DisciplineModel, calibrated_scales, game_state_summary, simulate_blocks
from src.market import asian_markets, score_matrix_from_samples
from src.names import normalize_team_name
from src.players import NEUTRAL_EFFECTS, PlayerRatings
from src.team_ratings import TeamUnitRatings, _logit

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
        discipline: DisciplineModel | None = None,
        game_state: bool = SIMULATION_CONFIG["game_state"],
        game_state_config: dict | None = None,
    ):
        self.ratings = ratings
        self.players = players
        self.n_sims = int(n_sims)
        self.performance_sd = float(performance_sd)
        self.seed = seed
        self.discipline = discipline or DisciplineModel()
        self.game_state = bool(game_state)
        self.game_state_config = {**GAME_STATE_CONFIG, **(game_state_config or {})}

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
        possession_shift: float = 0.0,
    ) -> dict[str, dict[str, float]]:
        """Per-side expected volume (chances or shots), on-target share and conversion after adjustments.

        `possession_shift` moves the home side's possession by that many percentage points
        (the away side loses the same); chance volume responds through the fitted possession
        elasticity. `midfield_shift` is a direct log-scale tilt of chance volume toward the home
        side (0.1 is roughly +10% home chances and -10% away chances).
        """
        base = self.ratings.matchup_rates(home, away)
        eff = {"home": self._effects(home, home_absent), "away": self._effects(away, away_absent)}
        shift = midfield_shift
        home_poss = base["home"].get("possession")
        if possession_shift:
            if home_poss is None:
                raise ValueError("possession_shift needs possession data (HPoss/APoss columns)")
            new_poss = float(np.clip(home_poss + possession_shift / 100.0, 0.05, 0.95))
            elasticity = self.ratings.possession_elasticity_ or 0.0
            # Elasticity links logit(possession) to log(home volume / away volume); split it across both sides.
            shift += 0.5 * elasticity * float(_logit(new_poss) - _logit(home_poss))
            home_poss = new_poss
        out = {}
        for side, opp, sign in (("home", "away", 1.0), ("away", "home", -1.0)):
            r, me, them = base[side], eff[side], eff[opp]
            scale = me["shots_for"] * them["shots_against"] * np.exp(sign * shift)
            volume = r["volume"] * scale
            p_on = float(np.clip(r["p_on_target"] * them["on_target_against"], 0.02, 0.95))
            p_conv = float(np.clip(r["p_convert"] * me["finishing"] * them["conversion_against"], 0.01, 0.95))
            out[side] = {
                "volume": float(volume),
                "expected_shots": float(r["expected_shots"] * scale),
                "p_on_target": p_on,
                "p_convert": p_conv,
                "expected_shots_on_target": float(volume * p_on),
                "xg": float(volume * p_on * p_conv),  # open play
            }
            if r.get("pen_rate"):
                out[side]["pen_rate"] = r["pen_rate"]
                out[side]["pen_conv"] = r["pen_conv"]
            out[side]["expected_goals"] = out[side]["xg"] + r.get("pen_rate", 0.0) * r.get("pen_conv", 0.0)
            if home_poss is not None:
                out[side]["possession"] = home_poss if side == "home" else 1.0 - home_poss
        return out

    # ------------------------------------------------------------- simulation
    def red_rates(self, home: str, away: str, referee: str | None = None) -> dict[str, float]:
        return {"home": self.discipline.red_rate(home, referee), "away": self.discipline.red_rate(away, referee)}

    def _draw(
        self, rates: dict[str, dict[str, float]], seed: int, red_rates: dict[str, float] | None = None
    ) -> dict[str, BlockDraws]:
        rng = np.random.default_rng(seed)
        if self.game_state:
            red = red_rates or {"home": self.discipline.league_rate_, "away": self.discipline.league_rate_}
            scales = calibrated_scales(
                rates, red, self.performance_sd, seed, self.game_state_config, n_sims=self.n_sims
            )
            return simulate_blocks(rates, red, self.n_sims, self.performance_sd, rng, self.game_state_config, scales)
        sd = self.performance_sd
        draws = {}
        for side in ("home", "away"):
            r = rates[side]
            noise = np.exp(rng.normal(-0.5 * sd**2, sd, self.n_sims))  # mean-one log-normal
            volume = rng.poisson(r["volume"] * noise)
            sot = rng.binomial(volume, r["p_on_target"])
            goals = rng.binomial(sot, r["p_convert"])
            if r.get("pen_rate"):
                goals = goals + rng.binomial(rng.poisson(r["pen_rate"], self.n_sims), r["pen_conv"])
            draws[side] = BlockDraws(volume, sot, goals, goals[:, None], np.zeros(self.n_sims, dtype=int))
        return draws

    def simulate(
        self,
        home: str,
        away: str,
        midfield_shift: float = 0.0,
        home_absent: list[str] | None = None,
        away_absent: list[str] | None = None,
        possession_shift: float = 0.0,
        referee: str | None = None,
    ) -> dict[str, Any]:
        """Runs `n_sims` matches and summarises attack, defence, midfield and scorer outcomes."""
        h, a = normalize_team_name(home), normalize_team_name(away)
        if h == a:
            raise ValueError("Home and away team must be different")
        rates = self.side_rates(h, a, midfield_shift, home_absent, away_absent, possession_shift)
        red = self.red_rates(h, a, referee)
        draws = self._draw(rates, self.seed, red)
        hg, ag = draws["home"].goals, draws["away"].goals

        teams = {"home": (h, a, True, home_absent), "away": (a, h, False, away_absent)}
        sides: dict[str, Any] = {}
        for side, (team, opp, at_home, absent) in teams.items():
            d = draws[side]
            volume, sot, goals = d.volume, d.sot, d.goals
            conceded = draws["away" if side == "home" else "home"].goals
            sides[side] = {
                "team": team,
                "attack": self._goal_summary(goals, volume, sot, rates[side]["expected_shots"]),
                "defence": self._defence_summary(team, opp, at_home, conceded),
                "goal_breakdown": self.ratings.goal_decomposition(team, opp, at_home).to_dict(orient="index"),
                "midfield_effect": self._midfield_effect(team, opp, rates[side]),
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
            "possession_shift": possession_shift,
            "volume_stat": self.ratings.volume_stat,
            "expected_possession": (
                {
                    "home": round(rates["home"]["possession"], 3),
                    "away": round(rates["away"]["possession"], 3),
                    "home_baseline": round(self.ratings.expected_possession(h, a), 3),
                }
                if "possession" in rates["home"]
                else None
            ),
            "outcome": {
                "home_win": round(float(np.mean(hg > ag)), 4),
                "draw": round(float(np.mean(hg == ag)), 4),
                "away_win": round(float(np.mean(hg < ag)), 4),
            },
            "top_scorelines": [
                {"score": f"{x}-{y}", "probability": round(float(p), 4)} for (x, y), p in scores.items()
            ],
            "unit_comparison": self.unit_comparison(h, a),
            "markets": {
                "over_2_5": round(float(np.mean(hg + ag > 2)), 4),
                "btts": round(float(np.mean((hg > 0) & (ag > 0))), 4),
                **asian_markets(score_matrix_from_samples(hg, ag)),
            },
            "game_state": (
                {
                    **game_state_summary(draws),
                    "red_card_rates": {k: round(v, 4) for k, v in red.items()},
                    "referee": referee,
                }
                if self.game_state
                else None
            ),
            "home": sides["home"],
            "away": sides["away"],
        }

    @staticmethod
    def _goal_summary(goals: np.ndarray, volume: np.ndarray, sot: np.ndarray, expected_shots: float) -> dict[str, Any]:
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
            "mean_volume": round(float(volume.mean()), 2),  # chances created (or shots; see volume_stat)
            "mean_shots_on_target": round(float(sot.mean()), 2),
            "p_3_plus_shots_on_target": round(float(np.mean(sot >= 3)), 4),
            "expected_shots": round(float(expected_shots), 2),
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

    def _midfield_effect(self, team: str, opp: str, rates: dict[str, float]) -> dict[str, Any]:
        """Goals gained or lost purely through the midfield battle (chances created, else shots)."""
        layer = self.ratings.layers_[self.ratings.volume_stat]
        volume_edge = layer.gen[team] + layer.allow[opp]
        neutral_xg = rates["xg"] / np.exp(volume_edge)
        out: dict[str, Any] = {
            "volume_stat": self.ratings.volume_stat,
            "volume_multiplier": round(float(np.exp(volume_edge)), 3),
            "xg_with_neutral_midfield": round(float(neutral_xg), 3),
            "goals_from_midfield_edge": round(float(rates["xg"] - neutral_xg), 3),
            "midfield_components": {
                k: round(float(np.exp(v)), 3) for k, v in self.ratings.midfield_components(team).items()
            },
        }
        if "possession" in rates:
            out["expected_possession"] = round(rates["possession"], 3)
        return out

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
    def score_matrix(self, home: str, away: str, referee: str | None = None, **what_if: Any) -> np.ndarray:
        """Simulated joint (home goals, away goals) distribution, for pricing any market."""
        h, a = normalize_team_name(home), normalize_team_name(away)
        draws = self._draw(self.side_rates(h, a, **what_if), self.seed, self.red_rates(h, a, referee))
        return score_matrix_from_samples(draws["home"].goals, draws["away"].goals)

    def unit_comparison(self, home: str, away: str) -> list[dict[str, Any]]:
        """Head-to-head unit battles (indices: 100 = league average)."""
        ph, pa = self.ratings.unit_profile(home), self.ratings.unit_profile(away)
        h, a = normalize_team_name(home), normalize_team_name(away)
        battles = [("Midfield control", h, "midfield", a, "midfield")]
        if "possession" in ph:
            battles.append(("Possession (expected % vs average side)", h, "possession", a, "possession"))
        if "chances_created" in ph:
            battles += [
                (f"{h} chances created vs {a} chance suppression", h, "chances_created", a, "chance_suppression"),
                (f"{a} chances created vs {h} chance suppression", a, "chances_created", h, "chance_suppression"),
            ]
        battles += [
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

    def midfield_sensitivity(self, home: str, away: str, shifts: tuple[float, ...] | None = None) -> pd.DataFrame:
        """How expected goals and results move as the midfield battle tilts (same random numbers per row).

        With possession data the rows are possession swings in percentage points for the home
        side; otherwise they are direct log-scale tilts of chance/shot volume.
        """
        by_possession = self.ratings.has_possession
        if shifts is None:
            shifts = (-10.0, -5.0, 0.0, 5.0, 10.0) if by_possession else (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
        vol = self.ratings.volume_stat
        rows = []
        for shift in shifts:
            kwargs = {"possession_shift": shift} if by_possession else {"midfield_shift": shift}
            rates = self.side_rates(home, away, **kwargs)
            draws = self._draw(rates, self.seed)
            hg, ag = draws["home"].goals, draws["away"].goals
            row: dict[str, Any] = {"possession_shift" if by_possession else "midfield_shift": shift}
            if by_possession:
                row["home_possession_pct"] = round(100.0 * rates["home"]["possession"], 1)
            rows.append(
                {
                    **row,
                    f"home_{vol}": round(rates["home"]["volume"], 1),
                    f"away_{vol}": round(rates["away"]["volume"], 1),
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
            xg = self.side_rates(home, away)["home" if at_home else "away"]["expected_goals"]
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
