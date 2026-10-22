"""Opponent-adjusted team unit ratings: midfield control, chance quality, finishing, defence.

Three time-decayed Poisson layers are fitted on the same matches:

    shots   ~ exp(c_s + home_s + gen_s[att] + allow_s[def])     -> midfield / territorial battle
    on tgt  ~ exp(c_t + home_t + gen_t[att] + allow_t[def])
    goals   ~ exp(c_g + home_g + gen_g[att] + allow_g[def])

Because each layer is a log-linear model, a team's goal threat telescopes exactly into
volume (shots) x chance quality (on-target share) x finishing (goals per shot on target),
which is what lets the simulator isolate how much of a matchup is decided in midfield.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import MODEL_CONFIG
from src.names import normalize_team_name

logger = logging.getLogger(__name__)

# layer name -> (home stat column, away stat column)
LAYERS = {"shots": ("HS", "AS"), "sot": ("HST", "AST"), "goals": ("FTHG", "FTAG")}


@dataclass
class PoissonLayer:
    """Fitted attack-generation / defence-allowance effects for one count statistic."""

    gen: dict[str, float]
    allow: dict[str, float]
    intercept: float
    home: float

    def rate(self, attacking: str, defending: str, at_home: bool) -> float:
        log_rate = self.intercept + self.gen.get(attacking, 0.0) + self.allow.get(defending, 0.0)
        return float(np.exp(log_rate + (self.home if at_home else 0.0)))


def fit_poisson_layer(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_y: np.ndarray,
    away_y: np.ndarray,
    weights: np.ndarray,
    n: int,
    l2: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Weighted Poisson regression with sum-to-zero team generation/allowance effects."""

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        gen = params[:n] - params[:n].mean()
        allow = params[n : 2 * n] - params[n : 2 * n].mean()
        c, h = params[2 * n], params[2 * n + 1]
        log_rh = c + h + gen[home_idx] + allow[away_idx]
        log_ra = c + gen[away_idx] + allow[home_idx]
        rh, ra = np.exp(log_rh), np.exp(log_ra)
        nll = -np.sum(weights * (home_y * log_rh - rh + away_y * log_ra - ra))
        nll += l2 * (np.sum(gen**2) + np.sum(allow**2))

        g_h = weights * (home_y - rh)
        g_a = weights * (away_y - ra)
        g_gen = np.bincount(home_idx, g_h, n) + np.bincount(away_idx, g_a, n)
        g_allow = np.bincount(away_idx, g_h, n) + np.bincount(home_idx, g_a, n)
        grad = np.empty_like(params)
        grad[:n] = -(g_gen - g_gen.mean()) + 2.0 * l2 * gen
        grad[n : 2 * n] = -(g_allow - g_allow.mean()) + 2.0 * l2 * allow
        grad[2 * n] = -(g_h.sum() + g_a.sum())
        grad[2 * n + 1] = -g_h.sum()
        return float(nll), grad

    mean_y = max(float(np.average(np.concatenate([home_y, away_y]))), 1e-3)
    x0 = np.concatenate([np.zeros(2 * n), [np.log(mean_y), 0.1]])
    bounds = [(-3.0, 3.0)] * (2 * n) + [(-5.0, 5.0), (-1.0, 1.0)]
    res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    gen = res.x[:n] - res.x[:n].mean()
    allow = res.x[n : 2 * n] - res.x[n : 2 * n].mean()
    return gen, allow, float(res.x[2 * n]), float(res.x[2 * n + 1])


class TeamUnitRatings:
    """Time-decayed, opponent-adjusted ratings for each team's attack, midfield and defence units."""

    def __init__(self, halflife_days: float = MODEL_CONFIG["dixon_coles_halflife_days"], l2: float = 0.01):
        self.halflife_days = float(halflife_days)
        self.l2 = float(l2)
        self.teams_: list[str] = []
        self.layers_: dict[str, PoissonLayer] = {}
        self.as_of_: pd.Timestamp | None = None

    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> TeamUnitRatings:
        missing = {c for cols in LAYERS.values() for c in cols} - set(df.columns)
        if missing:
            raise ValueError(f"Unit ratings need shot data; missing columns: {sorted(missing)}")
        data = df.copy()
        data["Date"] = pd.to_datetime(data["Date"])
        data["HomeTeam"] = data["HomeTeam"].map(normalize_team_name)
        data["AwayTeam"] = data["AwayTeam"].map(normalize_team_name)
        if as_of is not None:
            data = data[data["Date"] < pd.Timestamp(as_of)]
        if data.empty:
            raise ValueError("No matches available to fit unit ratings")

        ref = pd.Timestamp(as_of) if as_of is not None else data["Date"].max()
        days_ago = (ref - data["Date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
        all_weights = np.exp(-np.log(2.0) / self.halflife_days * days_ago)

        self.teams_ = sorted(set(data["HomeTeam"]).union(data["AwayTeam"]))
        idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)
        for name, (hcol, acol) in LAYERS.items():
            ok = data[[hcol, acol]].notna().all(axis=1).to_numpy()
            if ok.sum() < n:
                raise ValueError(f"Too few matches with {hcol}/{acol} to rate {n} teams")
            sub = data[ok]
            gen, allow, c, h = fit_poisson_layer(
                sub["HomeTeam"].map(idx).to_numpy(),
                sub["AwayTeam"].map(idx).to_numpy(),
                sub[hcol].to_numpy(dtype=float),
                sub[acol].to_numpy(dtype=float),
                all_weights[ok],
                n,
                self.l2,
            )
            self.layers_[name] = PoissonLayer(
                gen={t: float(gen[i]) for t, i in idx.items()},
                allow={t: float(allow[i]) for t, i in idx.items()},
                intercept=c,
                home=h,
            )
        self.as_of_ = ref
        return self

    def knows(self, team: str) -> bool:
        return normalize_team_name(team) in self.teams_

    def _effects(self, team: str) -> dict[str, float]:
        t = normalize_team_name(team)
        if t not in self.teams_:
            raise ValueError(f"Unknown team: {team!r}")
        s, o, g = self.layers_["shots"], self.layers_["sot"], self.layers_["goals"]
        return {
            "shots_gen": s.gen[t],
            "shots_allow": s.allow[t],
            "sot_gen": o.gen[t],
            "sot_allow": o.allow[t],
            "goals_gen": g.gen[t],
            "goals_allow": g.allow[t],
        }

    def unit_profile(self, team: str) -> dict[str, float]:
        """Unit indices where 100 = league average and higher is always better for the team.

        attack        goals scored vs an average defence
        defence       goals prevented vs an average attack
        midfield      net territorial control: shots won minus shots conceded
        chance_creation / shot_suppression   the two halves of the midfield battle
        chance_quality      share of own shots that hit the target
        finishing           goals per shot on target
        shot_quality_allowed  how wayward the shots it concedes are
        goalkeeping         goals prevented per shot on target faced
        """
        e = self._effects(team)

        def idx(log_effect: float) -> float:
            return round(100.0 * float(np.exp(log_effect)), 1)

        return {
            "attack": idx(e["goals_gen"]),
            "defence": idx(-e["goals_allow"]),
            "midfield": idx(0.5 * (e["shots_gen"] - e["shots_allow"])),
            "chance_creation": idx(e["shots_gen"]),
            "shot_suppression": idx(-e["shots_allow"]),
            "chance_quality": idx(e["sot_gen"] - e["shots_gen"]),
            "finishing": idx(e["goals_gen"] - e["sot_gen"]),
            "shot_quality_allowed": idx(-(e["sot_allow"] - e["shots_allow"])),
            "goalkeeping": idx(-(e["goals_allow"] - e["sot_allow"])),
        }

    def league_table(self) -> pd.DataFrame:
        """All teams' unit indices, strongest overall first."""
        table = pd.DataFrame({t: self.unit_profile(t) for t in self.teams_}).T
        table["overall"] = (np.sqrt(table["attack"] * table["defence"])).round(1)
        return table.sort_values("overall", ascending=False)

    def matchup_rates(self, home: str, away: str) -> dict[str, dict[str, float]]:
        """Expected shots, on-target share and conversion for each side of a fixture."""
        h, a = normalize_team_name(home), normalize_team_name(away)
        for t in (h, a):
            self._effects(t)
        out = {}
        for side, (att, dfn, at_home) in {"home": (h, a, True), "away": (a, h, False)}.items():
            shots = self.layers_["shots"].rate(att, dfn, at_home)
            sot = self.layers_["sot"].rate(att, dfn, at_home)
            goals = self.layers_["goals"].rate(att, dfn, at_home)
            out[side] = {
                "shots": shots,
                "p_on_target": float(np.clip(sot / shots, 0.02, 0.95)),
                "p_convert": float(np.clip(goals / sot, 0.01, 0.95)),
                "goals": goals,
            }
        return out

    def goal_decomposition(self, attacking: str, defending: str, at_home: bool) -> pd.DataFrame:
        """Splits the attacking side's expected goals into midfield, chance-quality and finishing effects.

        Each row shows the multiplicative effect (1.00 = league average) contributed by the
        attacking team's unit and by the opposing unit; the product of all rows times the
        league baseline equals the expected goals.
        """
        ea, ed = self._effects(attacking), self._effects(defending)
        rows = {
            "midfield / shot volume": (ea["shots_gen"], ed["shots_allow"]),
            "chance quality (on target %)": (ea["sot_gen"] - ea["shots_gen"], ed["sot_allow"] - ed["shots_allow"]),
            "finishing vs goalkeeping": (ea["goals_gen"] - ea["sot_gen"], ed["goals_allow"] - ed["sot_allow"]),
        }
        table = pd.DataFrame(
            {
                "attacking_unit": {k: np.exp(v[0]) for k, v in rows.items()},
                "opposing_unit": {k: np.exp(v[1]) for k, v in rows.items()},
            }
        )
        table["net_effect"] = table["attacking_unit"] * table["opposing_unit"]
        return table.round(3)


def season_timeline(
    df: pd.DataFrame,
    teams: list[str] | None = None,
    freq_days: int = 7,
    start: str | pd.Timestamp | None = None,
    min_matches: int = 60,
    **rating_kwargs: float,
) -> pd.DataFrame:
    """Walk-forward unit indices through time: each snapshot uses only matches before its date."""
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    first = pd.Timestamp(start) if start is not None else data["Date"].min()
    snapshots = pd.date_range(first, data["Date"].max() + pd.Timedelta(days=1), freq=f"{freq_days}D")
    wanted = {normalize_team_name(t) for t in teams} if teams else None
    rows = []
    for snap in snapshots:
        if (data["Date"] < snap).sum() < min_matches:
            continue
        model = TeamUnitRatings(**rating_kwargs).fit(data, as_of=snap)
        for team in model.teams_:
            if wanted is None or team in wanted:
                rows.append({"date": snap, "team": team, **model.unit_profile(team)})
    return pd.DataFrame(rows)
