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


# Hidden midfield control (territory / shot volume) and physicality traits, so the
# unit ratings and tactical profiles have something real to recover.
MIDFIELD: dict[str, float] = {
    "Manchester City": 0.30, "Arsenal": 0.22, "Liverpool": 0.15, "Brighton": 0.25, "Chelsea": 0.10,
    "Tottenham": 0.10, "Aston Villa": 0.05, "Newcastle United": 0.0, "Manchester United": 0.0,
    "Fulham": 0.05, "Southampton": -0.02, "Brentford": -0.05, "Bournemouth": -0.05, "West Ham United": -0.10,
    "Crystal Palace": -0.10, "Wolverhampton Wanderers": -0.10, "Leicester City": -0.10, "Ipswich Town": -0.15,
    "Everton": -0.20, "Nottingham Forest": -0.25,
}  # fmt: skip
AGGRESSION: dict[str, float] = {
    "Everton": 0.20, "Nottingham Forest": 0.15, "Wolverhampton Wanderers": 0.12, "Crystal Palace": 0.10,
    "Brentford": 0.08, "Manchester City": -0.15, "Brighton": -0.10, "Arsenal": -0.05,
}  # fmt: skip
POSITIONS = {"GK": 2, "DEF": 7, "MID": 7, "FWD": 4}
# Shot-placement skill (multiplies post-shot xG per unit of xG) and keeper quality
# (multiplies goals conceded per unit of post-shot xG faced; below 1 = good keeper).
FINISHING: dict[str, float] = {
    "Liverpool": 1.15, "Manchester City": 1.08, "Tottenham": 1.06, "Nottingham Forest": 1.10,
    "Brighton": 0.86, "Bournemouth": 0.90, "Everton": 0.92,
}  # fmt: skip
KEEPER: dict[str, float] = {
    "Brentford": 0.82, "Arsenal": 0.88, "Crystal Palace": 0.90, "Newcastle United": 0.92,
    "Southampton": 1.18, "Ipswich Town": 1.12, "Chelsea": 1.10,
}  # fmt: skip
REFEREES: dict[str, float] = {
    "A Taylor": 1.0, "M Oliver": 0.8, "S Attwell": 1.0, "P Tierney": 1.3, "C Kavanagh": 1.1,
    "R Jones": 0.9, "J Gillett": 1.0, "D Coote": 1.6,
}  # fmt: skip


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
    midfield = np.array([MIDFIELD.get(n, 0.0) for n in names])
    aggression = np.array([AGGRESSION.get(n, 0.0) for n in names])
    finishing = np.array([FINISHING.get(n, 1.0) for n in names])
    keeper = np.array([KEEPER.get(n, 1.0) for n in names])
    referees = list(REFEREES)

    rows: list[dict[str, Any]] = []
    start_date = pd.Timestamp("2023-08-12")
    goals = np.arange(11)

    for season in range(n_seasons):
        season_start = start_date + pd.Timedelta(days=364 * season)
        for rnd, fixtures in enumerate(round_robin_schedule(len(TEAMS), rng)):
            round_date = season_start + pd.Timedelta(weeks=rnd)
            for hi, ai in fixtures:
                match_date = round_date + pd.Timedelta(days=int(rng.integers(0, 3)))
                mid_edge = midfield[hi] - midfield[ai]
                lam = float(np.exp(attack[hi] + defence[ai] + home_adv + 0.5 * mid_edge))
                mu = float(np.exp(attack[ai] + defence[hi] - 0.5 * mid_edge))

                # Goals = open play (npxG -> post-shot xG via finishing -> goals via the opposing keeper)
                # plus penalties; lam / mu stay the total expected goals used for the odds.
                side_stats = {}
                for key, rate, att_i, def_i in (("H", lam, hi, ai), ("A", mu, ai, hi)):
                    pen_rate = 0.11 * np.exp(0.4 * (attack[att_i] + defence[def_i]))
                    np_rate = max(rate - 0.78 * pen_rate, 0.05)
                    npxg = np_rate / (finishing[att_i] * keeper[def_i]) * rng.gamma(10.0, 0.1)
                    psxg = npxg * finishing[att_i] * np.exp(rng.normal(-0.005, 0.1))
                    np_goals = int(rng.poisson(psxg * keeper[def_i]))
                    pk_att = int(rng.poisson(pen_rate))
                    pk = int(rng.binomial(pk_att, 0.78))
                    side_stats[key] = (np_goals + pk, round(npxg, 2), round(psxg, 2), pk_att, pk)
                fthg, ftag = side_stats["H"][0], side_stats["A"][0]
                ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
                referee = referees[int(rng.integers(len(referees)))]
                ref_factor = REFEREES[referee]

                # Bookmaker prices from the true rates plus noise, with a fixed overround
                mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
                true_p = np.array([np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()])
                # Opening prices are noisier than closing prices (the market learns before kick-off)
                book_p = true_p * np.exp(rng.normal(0.0, 0.06, 3))
                book_p = book_p / book_p.sum() * margin
                b365h, b365d, b365a = (round(float(1.0 / p), 2) for p in book_p)
                close_p = true_p * np.exp(rng.normal(0.0, 0.02, 3))
                close_p = close_p / close_p.sum() * 1.025
                psch, pscd, psca = (round(float(1.0 / p), 2) for p in close_p)

                # Midfield sets possession and chances created; an in-match possession swing
                # (poss_shock) moves chances too, so possession -> chances is a real, learnable effect.
                poss_shock = rng.normal(0.0, 0.3)
                home_poss = 100.0 / (1.0 + np.exp(-(0.08 + 1.4 * mid_edge + poss_shock)))
                cc_h = rng.poisson(
                    np.exp(np.log(9.5) + 0.06 + 0.8 * mid_edge + 0.3 * (attack[hi] + defence[ai]) + 0.8 * poss_shock)
                )
                cc_a = rng.poisson(
                    np.exp(np.log(8.0) - 0.8 * mid_edge + 0.3 * (attack[ai] + defence[hi]) - 0.8 * poss_shock)
                )
                # Shots = assisted chances plus some unassisted efforts
                shots_h = cc_h + rng.poisson(3.5)
                shots_a = cc_a + rng.poisson(3.0)
                hst = max(fthg, int(rng.binomial(shots_h, np.clip(3.0 * lam / max(shots_h, 1), 0.15, 0.6))))
                ast = max(ftag, int(rng.binomial(shots_a, np.clip(3.0 * mu / max(shots_a, 1), 0.15, 0.6))))
                rows.append(
                    {
                        "Date": match_date.strftime("%Y-%m-%d"),
                        "HomeTeam": names[hi],
                        "AwayTeam": names[ai],
                        "FTHG": fthg,
                        "FTAG": ftag,
                        "FTR": ftr,
                        "HS": max(int(shots_h), hst),
                        "AS": max(int(shots_a), ast),
                        "HST": hst,
                        "AST": ast,
                        "HPoss": round(home_poss, 1),
                        "APoss": round(100.0 - home_poss, 1),
                        "HCC": int(cc_h),
                        "ACC": int(cc_a),
                        "HC": int(rng.poisson(0.42 * max(shots_h, 1))),
                        "AC": int(rng.poisson(0.42 * max(shots_a, 1))),
                        "HF": int(rng.poisson(np.exp(np.log(10.5) + aggression[hi]))),
                        "AF": int(rng.poisson(np.exp(np.log(11.0) + aggression[ai]))),
                        "HY": int(rng.poisson(1.7 * np.exp(aggression[hi]) * ref_factor**0.5)),
                        "AY": int(rng.poisson(1.9 * np.exp(aggression[ai]) * ref_factor**0.5)),
                        "HR": int(rng.poisson(0.05 * np.exp(3 * aggression[hi]) * ref_factor)),
                        "AR": int(rng.poisson(0.05 * np.exp(3 * aggression[ai]) * ref_factor)),
                        "Referee": referee,
                        "HnpxG": side_stats["H"][1],
                        "AnpxG": side_stats["A"][1],
                        "HPSxG": side_stats["H"][2],
                        "APSxG": side_stats["A"][2],
                        "HPKatt": side_stats["H"][3],
                        "APKatt": side_stats["A"][3],
                        "HPK": side_stats["H"][4],
                        "APK": side_stats["A"][4],
                        "B365H": b365h,
                        "B365D": b365d,
                        "B365A": b365a,
                        "PSCH": psch,
                        "PSCD": pscd,
                        "PSCA": psca,
                    }
                )
            # Weekly random-walk drift in team strength (gives time-decay something to track)
            attack += rng.normal(0.0, drift_sd / np.sqrt(38), len(TEAMS))
            defence += rng.normal(0.0, drift_sd / np.sqrt(38), len(TEAMS))
            attack -= attack.mean()

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def _share(base: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalised share of a team total, given per-player base volume and positional weights."""
    raw = base * weights
    return raw / raw.sum()


def generate_synthetic_players(matches: pd.DataFrame, seed: int = 7, last_n_days: int = 365) -> pd.DataFrame:
    """Builds a synthetic player-season table whose totals match each team's recent match stats.

    Columns follow the player CSV schema used by src/players.py.
    """
    rng = np.random.default_rng(seed)
    data = matches[matches["Date"] >= matches["Date"].max() - pd.Timedelta(days=last_n_days)]
    rows: list[dict[str, Any]] = []
    shot_weight = {"GK": 0.0, "DEF": 0.25, "MID": 0.9, "FWD": 2.2}
    kp_weight = {"GK": 0.05, "DEF": 0.5, "MID": 2.0, "FWD": 1.0}
    def_weight = {"GK": 0.1, "DEF": 2.2, "MID": 1.6, "FWD": 0.4}

    for team in sorted(set(data["HomeTeam"]).union(data["AwayTeam"])):
        home, away = data[data["HomeTeam"] == team], data[data["AwayTeam"] == team]
        n_matches = len(home) + len(away)
        team_goals = int(home["FTHG"].sum() + away["FTAG"].sum())
        team_shots = int(home["HS"].sum() + away["AS"].sum())
        team_sot = int(home["HST"].sum() + away["AST"].sum())
        mid_trait = MIDFIELD.get(team, 0.0)

        squad = [(pos, k) for pos, count in POSITIONS.items() for k in range(1, count + 1)]
        # First-choice players (lower k) play most minutes; keepers rarely rotate.
        minutes = np.array(
            [
                n_matches * 90 * (0.95 if k == 1 else 0.05) if pos == "GK" else
                n_matches * 90 * float(np.clip(rng.normal(0.85 - 0.11 * (k - 1), 0.07), 0.05, 0.98))
                for pos, k in squad
            ]
        )  # fmt: skip
        quality = rng.lognormal(0.0, 0.35, len(squad))
        shots = rng.multinomial(team_shots, _share(minutes * quality, np.array([shot_weight[p] for p, _ in squad])))
        finishing = rng.lognormal(0.0, 0.3, len(squad))
        sot = np.minimum(shots, rng.multinomial(team_sot, (shots + 0.01) / (shots + 0.01).sum()))
        goals = np.minimum(sot, rng.multinomial(team_goals, (sot * finishing + 1e-3) / (sot * finishing + 1e-3).sum()))
        kp = rng.poisson(
            _share(minutes * quality, np.array([kp_weight[p] for p, _ in squad])) * team_shots * 0.7 * np.exp(mid_trait)
        )
        assists = rng.binomial(kp, 0.1)
        def_share = _share(minutes * quality, np.array([def_weight[p] for p, _ in squad]))
        tackles = rng.poisson(def_share * n_matches * 17 * np.exp(0.5 * mid_trait))
        interceptions = rng.poisson(def_share * n_matches * 9)

        for i, (pos, k) in enumerate(squad):
            rows.append(
                {
                    "player": f"{team} {pos}{k}",
                    "team": team,
                    "position": pos,
                    "minutes": int(minutes[i]),
                    "goals": int(goals[i]),
                    "assists": int(assists[i]),
                    "shots": int(shots[i]),
                    "shots_on_target": int(sot[i]),
                    "key_passes": int(kp[i]),
                    "tackles": int(tackles[i]),
                    "interceptions": int(interceptions[i]),
                }
            )
    return pd.DataFrame(rows)
