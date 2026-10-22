"""Opponent-adjusted team unit ratings: attack, midfield, defence.

Time-decayed Poisson layers are fitted on the same matches:

    chances ~ exp(c_c + home_c + gen_c[att] + allow_c[def])     (optional: HCC / ACC columns)
    shots   ~ exp(c_s + home_s + gen_s[att] + allow_s[def])
    on tgt  ~ exp(c_t + home_t + gen_t[att] + allow_t[def])
    goals   ~ exp(c_g + home_g + gen_g[att] + allow_g[def])

plus, when possession is available (HPoss / APoss), a logistic possession model

    logit(home possession share) = home_p + poss[home] - poss[away]

Ratings:
  * attack   blends goals and shots on target (on-target counts are far less noisy than goals)
  * midfield combines chances created won/conceded and possession share; it falls back to
    shot volume won/conceded when neither column is present
  * defence  goals prevented, with chance-quality and goalkeeping sub-indices

Because each layer is log-linear, a team's expected goals telescopes exactly into
volume (chances created, else shots) x chance quality (on target per chance) x finishing
(goals per shot on target), which is what lets the simulator isolate the midfield's share.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import MODEL_CONFIG, UNIT_CONFIG
from src.names import normalize_team_name

logger = logging.getLogger(__name__)

# layer name -> (home stat column, away stat column)
REQUIRED_LAYERS = {"shots": ("HS", "AS"), "sot": ("HST", "AST"), "goals": ("FTHG", "FTAG")}
OPTIONAL_LAYERS = {"chances": ("HCC", "ACC")}
POSSESSION_COLS = ("HPoss", "APoss")


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1.0 - p))


def possession_share(df: pd.DataFrame) -> pd.Series:
    """Home possession as a 0-1 share, accepting percentages or fractions and either column."""
    hcol, acol = POSSESSION_COLS
    home = pd.to_numeric(df[hcol], errors="coerce") if hcol in df else pd.Series(np.nan, index=df.index)
    if acol in df:
        away = pd.to_numeric(df[acol], errors="coerce")
        scale = 100.0 if np.nanmax(np.r_[home.to_numpy(dtype=float), away.to_numpy(dtype=float), 0.0]) > 1.0 else 1.0
        home = home.fillna(scale - away)
    else:
        scale = 100.0 if home.max(skipna=True) > 1.0 else 1.0
    return (home / scale).clip(0.01, 0.99)


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


@dataclass
class PossessionLayer:
    """Team possession strengths on the logit scale (0 = average side)."""

    strength: dict[str, float]
    home: float

    def expected_home_share(self, home: str, away: str) -> float:
        return _sigmoid(self.home + self.strength.get(home, 0.0) - self.strength.get(away, 0.0))


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


def fit_possession_layer(
    home_idx: np.ndarray, away_idx: np.ndarray, home_share: np.ndarray, weights: np.ndarray, n: int, l2: float = 0.5
) -> tuple[np.ndarray, float]:
    """Weighted ridge regression of logit(home possession) on home-minus-away team strengths."""
    m = len(home_idx)
    X = np.zeros((m, n + 1))
    X[np.arange(m), home_idx] += 1.0
    X[np.arange(m), away_idx] -= 1.0
    X[:, n] = 1.0  # home effect
    y = _logit(home_share)
    sw = np.sqrt(weights)
    ridge = np.sqrt(l2) * np.eye(n + 1)
    ridge[n, n] = 0.0  # do not shrink the home effect
    coef, *_ = np.linalg.lstsq(np.vstack([X * sw[:, None], ridge]), np.r_[y * sw, np.zeros(n + 1)], rcond=None)
    strength = coef[:n] - coef[:n].mean()
    return strength, float(coef[n])


class TeamUnitRatings:
    """Time-decayed, opponent-adjusted ratings for each team's attack, midfield and defence units."""

    def __init__(
        self,
        halflife_days: float = MODEL_CONFIG["dixon_coles_halflife_days"],
        l2: float = 0.01,
        attack_sot_weight: float = UNIT_CONFIG["attack_sot_weight"],
    ):
        self.halflife_days = float(halflife_days)
        self.l2 = float(l2)
        self.attack_sot_weight = float(attack_sot_weight)
        self.teams_: list[str] = []
        self.layers_: dict[str, PoissonLayer] = {}
        self.possession_: PossessionLayer | None = None
        self.possession_elasticity_: float | None = None
        self.as_of_: pd.Timestamp | None = None

    @property
    def volume_stat(self) -> str:
        """The first stage of the goal chain: chances created when available, else shots."""
        return "chances" if "chances" in self.layers_ else "shots"

    @property
    def has_possession(self) -> bool:
        return self.possession_ is not None

    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> TeamUnitRatings:
        missing = {c for cols in REQUIRED_LAYERS.values() for c in cols} - set(df.columns)
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
        h_idx = data["HomeTeam"].map(idx).to_numpy()
        a_idx = data["AwayTeam"].map(idx).to_numpy()

        self.layers_ = {}
        layers = {**REQUIRED_LAYERS, **{k: v for k, v in OPTIONAL_LAYERS.items() if set(v) <= set(data.columns)}}
        for name, (hcol, acol) in layers.items():
            values = data[[hcol, acol]].apply(pd.to_numeric, errors="coerce")
            ok = values.notna().all(axis=1).to_numpy()
            if ok.sum() < n:
                if name in REQUIRED_LAYERS:
                    raise ValueError(f"Too few matches with {hcol}/{acol} to rate {n} teams")
                continue
            gen, allow, c, h = fit_poisson_layer(
                h_idx[ok], a_idx[ok], values[hcol].to_numpy(float)[ok], values[acol].to_numpy(float)[ok],
                all_weights[ok], n, self.l2,
            )  # fmt: skip
            self.layers_[name] = PoissonLayer(
                gen={t: float(gen[i]) for t, i in idx.items()},
                allow={t: float(allow[i]) for t, i in idx.items()},
                intercept=c,
                home=h,
            )

        self.possession_ = None
        self.possession_elasticity_ = None
        if any(c in data.columns for c in POSSESSION_COLS):
            share = possession_share(data).to_numpy(dtype=float)
            ok = np.isfinite(share)
            if ok.sum() >= n:
                strength, home = fit_possession_layer(h_idx[ok], a_idx[ok], share[ok], all_weights[ok], n)
                self.possession_ = PossessionLayer({t: float(strength[i]) for t, i in idx.items()}, home)
                self.possession_elasticity_ = self._fit_possession_elasticity(data[ok], share[ok], all_weights[ok])
        self.as_of_ = ref
        return self

    def _fit_possession_elasticity(self, data: pd.DataFrame, share: np.ndarray, weights: np.ndarray) -> float:
        """How much the chance (or shot) ratio moves when a team gets more possession than expected.

        Regresses the match-level surprise in log(home volume / away volume) on the surprise in
        logit(home possession), so team quality (which drives both) is held fixed.
        """
        vol = self.layers_[self.volume_stat]
        hcol, acol = (OPTIONAL_LAYERS | REQUIRED_LAYERS)[self.volume_stat]
        hv = pd.to_numeric(data[hcol], errors="coerce").to_numpy(float)
        av = pd.to_numeric(data[acol], errors="coerce").to_numpy(float)
        exp_ratio = np.array(
            [np.log(vol.rate(h, a, True) / vol.rate(a, h, False)) for h, a in zip(data["HomeTeam"], data["AwayTeam"])]
        )
        exp_logit = np.array(
            [_logit(self.possession_.expected_home_share(h, a)) for h, a in zip(data["HomeTeam"], data["AwayTeam"])]  # type: ignore[union-attr]
        )
        r_vol = np.log((hv + 0.5) / (av + 0.5)) - exp_ratio
        r_pos = _logit(share) - exp_logit
        ok = np.isfinite(r_vol) & np.isfinite(r_pos)
        denom = np.sum(weights[ok] * r_pos[ok] ** 2)
        return float(np.sum(weights[ok] * r_vol[ok] * r_pos[ok]) / denom) if denom > 0 else 0.0

    def knows(self, team: str) -> bool:
        return normalize_team_name(team) in self.teams_

    def _resolve(self, team: str) -> str:
        t = normalize_team_name(team)
        if t not in self.teams_:
            raise ValueError(f"Unknown team: {team!r}")
        return t

    def _effects(self, team: str) -> dict[str, float]:
        t = self._resolve(team)
        out = {}
        for name, layer in self.layers_.items():
            out[f"{name}_gen"] = layer.gen[t]
            out[f"{name}_allow"] = layer.allow[t]
        out["vol_gen"] = out[f"{self.volume_stat}_gen"]
        out["vol_allow"] = out[f"{self.volume_stat}_allow"]
        if self.possession_ is not None:
            out["possession"] = self.possession_.strength[t]
        return out

    def midfield_components(self, team: str) -> dict[str, float]:
        """Log-scale midfield components (0 = average) that make up the midfield index."""
        e = self._effects(team)
        comps = {}
        if "chances_gen" in e:
            comps["chances_created_balance"] = 0.5 * (e["chances_gen"] - e["chances_allow"])
        if "possession" in e:
            comps["possession"] = e["possession"]
        if not comps:
            comps["shot_volume_balance"] = 0.5 * (e["shots_gen"] - e["shots_allow"])
        return comps

    def unit_profile(self, team: str) -> dict[str, float]:
        """Unit indices where 100 = league average and higher is always better for the team.

        attack            goals and shots on target vs an average defence (blend set by attack_sot_weight)
        goal_threat       goals alone;  shots_on_target   on-target shots alone
        defence           goals prevented vs an average attack
        midfield          chances created won-vs-conceded and possession share (shot volume if neither exists)
        possession        expected possession % against an average side at a neutral venue
        chances_created / chance_suppression   the two halves of the chance battle (if HCC/ACC exist)
        shot_volume / shot_suppression         shots won and shots prevented
        chance_quality    shots on target per chance (or per shot)
        finishing         goals per shot on target
        shot_quality_allowed  how wayward the chances it concedes are
        goalkeeping       goals prevented per shot on target faced
        """
        e = self._effects(team)
        w = self.attack_sot_weight

        def idx(log_effect: float) -> float:
            return round(100.0 * float(np.exp(log_effect)), 1)

        profile = {
            "attack": idx((1.0 - w) * e["goals_gen"] + w * e["sot_gen"]),
            "goal_threat": idx(e["goals_gen"]),
            "shots_on_target": idx(e["sot_gen"]),
            "defence": idx(-e["goals_allow"]),
            "midfield": idx(float(np.mean(list(self.midfield_components(team).values())))),
        }
        if "possession" in e:
            profile["possession"] = round(100.0 * _sigmoid(e["possession"]), 1)
        if "chances_gen" in e:
            profile["chances_created"] = idx(e["chances_gen"])
            profile["chance_suppression"] = idx(-e["chances_allow"])
        profile.update(
            {
                "shot_volume": idx(e["shots_gen"]),
                "shot_suppression": idx(-e["shots_allow"]),
                "chance_quality": idx(e["sot_gen"] - e["vol_gen"]),
                "finishing": idx(e["goals_gen"] - e["sot_gen"]),
                "shot_quality_allowed": idx(-(e["sot_allow"] - e["vol_allow"])),
                "goalkeeping": idx(-(e["goals_allow"] - e["sot_allow"])),
            }
        )
        return profile

    def league_table(self) -> pd.DataFrame:
        """All teams' unit indices, strongest overall first."""
        table = pd.DataFrame({t: self.unit_profile(t) for t in self.teams_}).T
        table["overall"] = (np.sqrt(table["attack"] * table["defence"])).round(1)
        return table.sort_values("overall", ascending=False)

    def expected_possession(self, home: str, away: str) -> float | None:
        """Expected home possession share (0-1), or None without possession data."""
        if self.possession_ is None:
            return None
        return self.possession_.expected_home_share(self._resolve(home), self._resolve(away))

    def matchup_rates(self, home: str, away: str) -> dict[str, dict[str, float]]:
        """Expected volume (chances or shots), on-target share and conversion for each side."""
        h, a = self._resolve(home), self._resolve(away)
        poss_home = self.expected_possession(h, a)
        vol_layer = self.layers_[self.volume_stat]
        out = {}
        for side, (att, dfn, at_home) in {"home": (h, a, True), "away": (a, h, False)}.items():
            volume = vol_layer.rate(att, dfn, at_home)
            sot = self.layers_["sot"].rate(att, dfn, at_home)
            goals = self.layers_["goals"].rate(att, dfn, at_home)
            out[side] = {
                "volume": volume,
                "p_on_target": float(np.clip(sot / volume, 0.02, 0.95)),
                "p_convert": float(np.clip(goals / sot, 0.01, 0.95)),
                "goals": goals,
                "expected_shots": self.layers_["shots"].rate(att, dfn, at_home),
            }
            if poss_home is not None:
                out[side]["possession"] = poss_home if side == "home" else 1.0 - poss_home
        return out

    def goal_decomposition(self, attacking: str, defending: str, at_home: bool) -> pd.DataFrame:
        """Splits the attacking side's expected goals into midfield, chance-quality and finishing effects.

        Each row shows the multiplicative effect (1.00 = league average) contributed by the
        attacking team's unit and by the opposing unit; the product of all rows times the
        league baseline equals the expected goals.
        """
        ea, ed = self._effects(attacking), self._effects(defending)
        volume_label = "midfield / chances created" if self.volume_stat == "chances" else "midfield / shot volume"
        rows = {
            volume_label: (ea["vol_gen"], ed["vol_allow"]),
            "chance quality (shots on target)": (ea["sot_gen"] - ea["vol_gen"], ed["sot_allow"] - ed["vol_allow"]),
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
