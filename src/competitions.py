"""Competitions beyond the domestic league: the UEFA Champions League.

Teams from different leagues can only be rated on one common scale when matches link
them, so Champions League fixtures are part of the match data (with a `Competition`
column). Ratings are then fitted jointly: every index stays "100 = average of all teams
in the data", and Bayern Munich vs Arsenal is a meaningful comparison.

The team list is the 2025/26 league phase (36 clubs). Strengths are demo values for the
synthetic data on the same log scale as the Premier League teams in src/synthetic.py:
(club, country, attack, defence); lower defence concedes fewer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DOMESTIC_LEAGUE = "Premier League"
CHAMPIONS_LEAGUE = "Champions League"

CHAMPIONS_LEAGUE_2025_26: list[tuple[str, str, float, float]] = [
    # England (also in the domestic demo league; their league strengths are used)
    ("Liverpool", "ENG", 0.0, 0.0),
    ("Arsenal", "ENG", 0.0, 0.0),
    ("Manchester City", "ENG", 0.0, 0.0),
    ("Chelsea", "ENG", 0.0, 0.0),
    ("Newcastle United", "ENG", 0.0, 0.0),
    ("Tottenham", "ENG", 0.0, 0.0),
    # Spain
    ("Real Madrid", "ESP", 0.42, -0.30),
    ("Barcelona", "ESP", 0.45, -0.18),
    ("Atletico Madrid", "ESP", 0.18, -0.22),
    ("Athletic Club", "ESP", 0.05, -0.08),
    ("Villarreal", "ESP", 0.08, 0.02),
    # Germany
    ("Bayern Munich", "GER", 0.48, -0.22),
    ("Bayer Leverkusen", "GER", 0.20, -0.08),
    ("Borussia Dortmund", "GER", 0.18, 0.00),
    ("Eintracht Frankfurt", "GER", 0.08, 0.08),
    # Italy
    ("Inter", "ITA", 0.30, -0.30),
    ("Napoli", "ITA", 0.18, -0.15),
    ("Juventus", "ITA", 0.12, -0.15),
    ("Atalanta", "ITA", 0.15, -0.05),
    # France
    ("Paris Saint-Germain", "FRA", 0.42, -0.28),
    ("Marseille", "FRA", 0.06, 0.04),
    ("Monaco", "FRA", 0.08, 0.05),
    # Portugal, Netherlands, Belgium
    ("Benfica", "POR", 0.10, -0.05),
    ("Sporting CP", "POR", 0.12, -0.02),
    ("PSV", "NED", 0.05, 0.05),
    ("Ajax", "NED", -0.02, 0.08),
    ("Club Brugge", "BEL", -0.05, 0.08),
    ("Union Saint-Gilloise", "BEL", -0.15, 0.12),
    # Rest of Europe
    ("Galatasaray", "TUR", -0.02, 0.10),
    ("Olympiacos", "GRE", -0.12, 0.12),
    ("Slavia Prague", "CZE", -0.15, 0.12),
    ("Copenhagen", "DEN", -0.18, 0.15),
    ("Bodo/Glimt", "NOR", -0.15, 0.18),
    ("Pafos", "CYP", -0.40, 0.35),
    ("Qarabag", "AZE", -0.35, 0.32),
    ("Kairat Almaty", "KAZ", -0.45, 0.40),
]

TEAM_COUNTRY = {name: country for name, country, _, _ in CHAMPIONS_LEAGUE_2025_26}
LEAGUE_PHASE_ROUNDS = 8


def champions_league_teams() -> list[str]:
    return [name for name, *_ in CHAMPIONS_LEAGUE_2025_26]


def league_phase_pairings(teams: list[int], rng: np.random.Generator, rounds: int = LEAGUE_PHASE_ROUNDS):
    """Swiss-style league phase: `rounds` matchdays, no repeated opponents, 4 home / 4 away each.

    Builds an 8-regular opponent graph from random perfect matchings, then orients every
    edge along an Euler circuit so each club gets exactly as many home as away games.
    """
    n = len(teams)
    if n % 2 or rounds % 2:
        raise ValueError("league phase needs an even number of teams and rounds")
    for _attempt in range(200):
        used: set[frozenset[int]] = set()
        matchdays: list[list[tuple[int, int]]] = []
        ok = True
        for _ in range(rounds):
            pairs = _random_matching(teams, used, rng)
            if pairs is None:
                ok = False
                break
            used.update(frozenset(p) for p in pairs)
            matchdays.append(pairs)
        if ok:
            orientation = _euler_orientation([p for md in matchdays for p in md])
            return [[p if p in orientation else (p[1], p[0]) for p in md] for md in matchdays]
    raise RuntimeError("could not build a league-phase schedule")


def _random_matching(teams: list[int], used: set[frozenset[int]], rng: np.random.Generator):
    for _ in range(200):
        pool = list(rng.permutation(teams))
        pairs = []
        while pool:
            a = pool.pop(0)
            partners = [b for b in pool if frozenset((a, b)) not in used]
            if not partners:
                break
            b = partners[0]
            pool.remove(b)
            pairs.append((int(a), int(b)))
        if not pool and len(pairs) == len(teams) // 2:
            return pairs
    return None


def _euler_orientation(edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Orients edges of an even-degree multigraph so in-degree equals out-degree (Hierholzer)."""
    adj: dict[int, list[tuple[int, int]]] = {}
    for i, (a, b) in enumerate(edges):
        adj.setdefault(a, []).append((b, i))
        adj.setdefault(b, []).append((a, i))
    used = [False] * len(edges)
    oriented: set[tuple[int, int]] = set()
    for start in adj:
        stack = [start]
        while stack:
            v = stack[-1]
            while adj[v] and used[adj[v][-1][1]]:
                adj[v].pop()
            if adj[v]:
                w, i = adj[v].pop()
                used[i] = True
                oriented.add((v, w))
                stack.append(w)
            else:
                stack.pop()
    return oriented


def league_phase_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Standings from league-phase results: points, then goal difference, then goals scored."""
    rows = []
    for r in matches.itertuples():
        for team, gf, ga in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            rows.append({"team": team, "pts": 3 if gf > ga else 1 if gf == ga else 0, "gd": gf - ga, "gf": gf})
    table = pd.DataFrame(rows).groupby("team").sum()
    return table.sort_values(["pts", "gd", "gf"], ascending=False)
