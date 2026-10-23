"""Time-decayed Dixon-Coles bivariate Poisson goal model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.config import MODEL_CONFIG, UNIT_CONFIG
from src.context import latest_change_before
from src.names import normalize_team_name

logger = logging.getLogger(__name__)


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles (1997) correlation adjustment for low-scoring outcomes (0-0, 1-0, 0-1, 1-1)."""
    if x == 0 and y == 0:
        return 1.0 - (lam * mu * rho)
    if x == 0 and y == 1:
        return 1.0 + (lam * rho)
    if x == 1 and y == 0:
        return 1.0 + (mu * rho)
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _make_objective(
    h_idx: np.ndarray,
    a_idx: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    n: int,
    l2: float,
):
    """Builds the weighted Dixon-Coles negative log-likelihood with an analytic gradient."""
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        """Weighted negative log-likelihood and its analytic gradient."""
        att = params[:n] - np.mean(params[:n])
        dfn = params[n : 2 * n]
        h_adv = params[2 * n]
        rho = params[2 * n + 1]

        log_lam = att[h_idx] + dfn[a_idx] + h_adv
        log_mu = att[a_idx] + dfn[h_idx]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        clipped = tau < 1e-6
        tau = np.where(clipped, 1e-6, tau)

        # Poisson log-pmf up to the constant -log(x!) terms
        ll = np.log(tau) + hg * log_lam - lam + ag * log_mu - mu
        nll = -np.sum(weights * ll) + l2 * (np.sum(att**2) + np.sum(dfn**2))

        # d log(tau) / d(log_lam, log_mu, rho)
        dtau_lam = np.zeros_like(lam)
        dtau_mu = np.zeros_like(lam)
        dtau_rho = np.zeros_like(lam)
        dtau_lam[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
        dtau_mu[m00] = dtau_lam[m00]
        dtau_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
        dtau_lam[m01] = lam[m01] * rho / tau[m01]
        dtau_rho[m01] = lam[m01] / tau[m01]
        dtau_mu[m10] = mu[m10] * rho / tau[m10]
        dtau_rho[m10] = mu[m10] / tau[m10]
        dtau_rho[m11] = -1.0 / tau[m11]
        for arr in (dtau_lam, dtau_mu, dtau_rho):
            arr[clipped] = 0.0

        g_lam = weights * (hg - lam + dtau_lam)
        g_mu = weights * (ag - mu + dtau_mu)
        g_att = np.bincount(h_idx, g_lam, n) + np.bincount(a_idx, g_mu, n)
        g_def = np.bincount(a_idx, g_lam, n) + np.bincount(h_idx, g_mu, n)

        grad = np.empty_like(params)
        # Attack params are mean-centred, so project the gradient onto the centred subspace.
        grad[:n] = -(g_att - g_att.mean()) + 2.0 * l2 * att
        grad[n : 2 * n] = -g_def + 2.0 * l2 * dfn
        grad[2 * n] = -np.sum(g_lam)
        grad[2 * n + 1] = -np.sum(weights * dtau_rho)
        return float(nll), grad

    return objective


class TimeDecayedDixonColes:
    """Bivariate Poisson goal model with Dixon-Coles low-score adjustment and exponential time-decay."""

    def __init__(
        self,
        halflife_days: float = MODEL_CONFIG["dixon_coles_halflife_days"],
        max_goals: int = MODEL_CONFIG["dixon_coles_max_goals"],
        l2: float = MODEL_CONFIG["dixon_coles_l2"],
    ):
        self.halflife_days = float(halflife_days)
        self.max_goals = int(max_goals)
        self.l2 = float(l2)
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_: float = 0.24
        self.rho_: float = -0.06

    def fit(
        self,
        df: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
        manager_changes: dict[str, pd.Timestamp] | None = None,
    ) -> TimeDecayedDixonColes:
        """Fits team strengths, weighting each match by exp(-xi * days before `as_of`).

        `as_of` defaults to the most recent match date in `df`.
        """
        req = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        missing = set(req) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for Dixon-Coles fit: {missing}")

        data = df.dropna(subset=req).copy()
        if data.empty:
            raise ValueError("Cannot fit Dixon-Coles on an empty dataset")
        data["Date"] = pd.to_datetime(data["Date"])
        data["HomeTeam"] = data["HomeTeam"].map(normalize_team_name)
        data["AwayTeam"] = data["AwayTeam"].map(normalize_team_name)

        ref_date = pd.Timestamp(as_of) if as_of is not None else data["Date"].max()
        days_ago = (ref_date - data["Date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
        weights = np.exp(-np.log(2.0) / self.halflife_days * days_ago)
        # Matches before a team's latest manager appointment say less about the team now
        for team, changed in (manager_changes or {}).items():
            before = ((data["HomeTeam"] == team) | (data["AwayTeam"] == team)) & (data["Date"] < pd.Timestamp(changed))
            weights = np.where(before.to_numpy(), weights * UNIT_CONFIG["manager_change_discount"], weights)

        teams = sorted(set(data["HomeTeam"]).union(data["AwayTeam"]))
        n = len(teams)
        t_idx = {t: i for i, t in enumerate(teams)}
        h_idx = data["HomeTeam"].map(t_idx).to_numpy()
        a_idx = data["AwayTeam"].map(t_idx).to_numpy()
        hg = data["FTHG"].astype(int).to_numpy()
        ag = data["FTAG"].astype(int).to_numpy()

        objective = _make_objective(h_idx, a_idx, hg, ag, weights, n, self.l2)
        x0 = np.concatenate([np.zeros(2 * n), [0.24, -0.05]])
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-0.2, 0.8), (-0.3, 0.3)]
        res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            logger.warning("Dixon-Coles optimiser did not fully converge: %s", res.message)

        att = res.x[:n] - np.mean(res.x[:n])
        dfn = res.x[n : 2 * n]
        self.teams_ = teams
        self.attack_ = {t: float(att[i]) for i, t in enumerate(teams)}
        self.defense_ = {t: float(dfn[i]) for i, t in enumerate(teams)}
        self.home_adv_ = float(res.x[2 * n])
        self.rho_ = float(res.x[2 * n + 1])
        return self

    def knows(self, team: str) -> bool:
        return normalize_team_name(team) in self.attack_

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals (lambda, mu). Unseen teams fall back to league-average strength."""
        h = normalize_team_name(home_team)
        a = normalize_team_name(away_team)
        lam = np.exp(self.attack_.get(h, 0.0) + self.defense_.get(a, 0.0) + self.home_adv_)
        mu = np.exp(self.attack_.get(a, 0.0) + self.defense_.get(h, 0.0))
        return float(lam), float(mu)

    def predict_score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        lam, mu = self.expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1)
        mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        for x in (0, 1):
            for y in (0, 1):
                mat[x, y] = max(0.0, mat[x, y] * _dc_tau(x, y, lam, mu, self.rho_))
        return mat / mat.sum()

    def predict_1x2(self, home_team: str, away_team: str) -> dict[str, float]:
        mat = self.predict_score_matrix(home_team, away_team)
        return {
            "H": float(np.sum(np.tril(mat, -1))),
            "D": float(np.sum(np.diag(mat))),
            "A": float(np.sum(np.triu(mat, 1))),
        }


def walk_forward_dc_probs(
    df: pd.DataFrame,
    refit_days: int = MODEL_CONFIG["dixon_coles_refit_days"],
    min_matches: int = MODEL_CONFIG["dixon_coles_min_matches"],
    manager_changes: dict[str, list[pd.Timestamp]] | None = None,
    **model_kwargs: float,
) -> np.ndarray:
    """Out-of-sample Dixon-Coles 1X2 probabilities for every row of `df`.

    The model is refit every `refit_days` using only matches played strictly
    before the refit date, so no row's probability ever depends on its own (or
    any later) result. Rows before `min_matches` of history get NaN.
    """
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    order = np.argsort(data["Date"].to_numpy(), kind="stable")
    dates = data["Date"].to_numpy()[order]
    probs = np.full((len(data), 3), np.nan)

    model: TimeDecayedDixonColes | None = None
    next_refit: np.datetime64 | None = None
    for pos, row_i in enumerate(order):
        dt = dates[pos]
        if next_refit is None or dt >= next_refit:
            history = data.iloc[order[: np.searchsorted(dates, dt, side="left")]]
            if len(history) >= min_matches:
                changes = latest_change_before(manager_changes or {}, pd.Timestamp(dt))
                model = TimeDecayedDixonColes(**model_kwargs).fit(
                    history, as_of=pd.Timestamp(dt), manager_changes=changes
                )
                next_refit = dt + np.timedelta64(refit_days, "D")
        if model is not None:
            p = model.predict_1x2(data.iloc[row_i]["HomeTeam"], data.iloc[row_i]["AwayTeam"])
            probs[row_i] = (p["H"], p["D"], p["A"])
    return probs
