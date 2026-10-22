"""Synthetic Premier League-style match generator for demos and tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

# (team, attack, defence) log-strengths; lower defence = concedes fewer
TEAMS: list[tuple[str, float, float]] = [
    ("Manchester City", 0.45, -0.35),
    ("Arsenal", 0.38, -0.30),
    ("Liverpool", 0.40, -0.25),
    ("Chelsea", 0.20, -0.10),
    ("Newcastle United", 0.18, -0.12),
    ("Tottenham", 0.25, -0.05),
    ("Aston Villa", 0.16, -0.08),
    ("Manchester United", 0.12, -0.05),
    ("Brighton", 0.10, -0.02),
    ("West Ham United", 0.02, 0.05),
    ("Fulham", -0.04, 0.04),
    ("Brentford", -0.02, 0.06),
    ("Crystal Palace", -0.08, 0.02),
    ("Bournemouth", -0.06, 0.08),
    ("Wolverhampton Wanderers", -0.12, 0.10),
    ("Everton", -0.15, 0.05),
    ("Nottingham Forest", -0.14, 0.12),
    ("Leicester City", -0.18, 0.18),
    ("Ipswich Town", -0.25, 0.24),
    ("Southampton", -0.28, 0.28),
]


def round_robin_schedule(n_teams: int, rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Double round-robin via the circle method: 2*(n-1) rounds, each team plays once per round."""
    if n_teams % 2:
        raise ValueError("round_robin_schedule needs an even number of teams")
    order = list(rng.permutation(n_teams))
    first_half: list[list[tuple[int, int]]] = []
    for rnd in range(n_teams - 1):
        pairs = []
        for k in range(n_teams // 2):
            a, b = order[k], order[n_teams - 1 - k]
            # Alternate venues so nobody is stuck at home/away for long runs
            pairs.append((a, b) if (rnd + k) % 2 == 0 else (b, a))
        first_half.append(pairs)
        order = [order[0], order[-1], *order[1:-1]]
    second_half = [[(b, a) for a, b in rnd] for rnd in first_half]
    return first_half + second_half


def generate_synthetic_matches(
    n_seasons: int = 3,
    seed: int = 42,
    home_adv: float = 0.22,
    drift_sd: float = 0.04,
    margin: float = 1.055,
) -> pd.DataFrame:
    """Generates realistic league match data with drifting team strengths and bookmaker odds."""
    rng = np.random.default_rng(seed)
    names = [t[0] for t in TEAMS]
    attack = np.array([t[1] for t in TEAMS], dtype=float)
    defence = np.array([t[2] for t in TEAMS], dtype=float)

    rows: list[dict[str, Any]] = []
    start_date = pd.Timestamp("2023-08-12")
    goals = np.arange(11)

    for season in range(n_seasons):
        season_start = start_date + pd.Timedelta(days=364 * season)
        for rnd, fixtures in enumerate(round_robin_schedule(len(TEAMS), rng)):
            round_date = season_start + pd.Timedelta(weeks=rnd)
            for hi, ai in fixtures:
                match_date = round_date + pd.Timedelta(days=int(rng.integers(0, 3)))
                lam = float(np.exp(attack[hi] + defence[ai] + home_adv))
                mu = float(np.exp(attack[ai] + defence[hi]))

                fthg = int(rng.poisson(lam))
                ftag = int(rng.poisson(mu))
                ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")

                # Bookmaker prices from the true rates plus noise, with a fixed overround
                mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
                true_p = np.array([np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()])
                book_p = true_p * np.exp(rng.normal(0.0, 0.04, 3))
                book_p = book_p / book_p.sum() * margin
                b365h, b365d, b365a = (round(float(1.0 / p), 2) for p in book_p)

                hst = max(fthg, int(rng.poisson(lam * 3.1)))
                ast = max(ftag, int(rng.poisson(mu * 2.9)))
                rows.append(
                    {
                        "Date": match_date.strftime("%Y-%m-%d"),
                        "HomeTeam": names[hi],
                        "AwayTeam": names[ai],
                        "FTHG": fthg,
                        "FTAG": ftag,
                        "FTR": ftr,
                        "HS": hst + int(rng.poisson(7.5)),
                        "AS": ast + int(rng.poisson(6.5)),
                        "HST": hst,
                        "AST": ast,
                        "B365H": b365h,
                        "B365D": b365d,
                        "B365A": b365a,
                    }
                )
            # Weekly random-walk drift in team strength (gives time-decay something to track)
            attack += rng.normal(0.0, drift_sd / np.sqrt(38), len(TEAMS))
            defence += rng.normal(0.0, drift_sd / np.sqrt(38), len(TEAMS))
            attack -= attack.mean()

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date", kind="stable").reset_index(drop=True)
