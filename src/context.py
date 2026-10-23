"""Competition context: league-table stakes, fixture congestion, season boundaries, manager changes."""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import pandas as pd

from src.names import normalize_team_name

SEASON_GAP_DAYS = 45  # a break longer than this between league matches starts a new season
EUROPE_SLOTS = 4  # league position that qualifies for the Champions League
DROP_SLOTS = 3  # relegation places


class LeagueTable:
    """Running league table for the current season, updated strictly after each result."""

    def __init__(self, expected_teams: int = 20):
        self.expected_teams = expected_teams
        self.points: dict[str, int] = {}
        self.played: dict[str, int] = {}
        self.goal_diff: dict[str, int] = {}

    def reset(self, expected_teams: int | None = None) -> None:
        if expected_teams:
            self.expected_teams = expected_teams
        self.points.clear()
        self.played.clear()
        self.goal_diff.clear()

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        for team, gf, ga in ((home, hg, ag), (away, ag, hg)):
            self.points[team] = self.points.get(team, 0) + (3 if gf > ga else 1 if gf == ga else 0)
            self.played[team] = self.played.get(team, 0) + 1
            self.goal_diff[team] = self.goal_diff.get(team, 0) + gf - ga

    @property
    def n_teams(self) -> int:
        return max(len(self.points), self.expected_teams)

    def _sorted_points(self) -> list[int]:
        pts = sorted(self.points.values(), reverse=True)
        return pts + [0] * (self.n_teams - len(pts))

    def stakes(self, team: str) -> dict[str, float]:
        """Pre-match motivation features for `team` from the current table."""
        games_per_season = 2 * (self.n_teams - 1)
        played = self.played.get(team, 0)
        left = max(games_per_season - played, 0)
        pts = self.points.get(team, 0)
        table = self._sorted_points()
        top4 = table[EUROPE_SLOTS - 1]
        safety = table[self.n_teams - DROP_SLOTS - 1]  # points of the last safe place (17th of 20)
        leader = table[0]
        # Dead rubber: cannot reach the European places or the title, and the drop cannot catch them
        can_reach_top4 = pts + 3 * left >= top4
        safe = pts - safety > 3 * left
        return {
            "games_left": float(left),
            "pts_to_top4": float(top4 - pts),
            "drop_margin": float(pts - safety),
            "pts_to_leader": float(leader - pts),
            "dead_rubber": float(played > 0 and left > 0 and not can_reach_top4 and safe),
            # How close the nearest boundary (title, top four, drop zone) is per game left: small = high stakes
            "stake_gap": float(min(abs(leader - pts), abs(top4 - pts), abs(pts - safety)) / (left + 1)),
            "points": float(pts),
        }


class FixtureCalendar:
    """All-competition fixture dates per team (league, cups, Europe) for rest and congestion features.

    Expects a CSV/DataFrame with columns Date, Team and optionally Competition. League matches
    passed to the feature builder are added automatically, so the calendar only needs the
    other competitions.
    """

    def __init__(self, fixtures: pd.DataFrame | None = None):
        self.dates: dict[str, list[pd.Timestamp]] = {}
        if fixtures is not None and not fixtures.empty:
            df = fixtures.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            for team, grp in df.groupby(df["Team"].map(normalize_team_name)):
                self.dates[team] = sorted(grp["Date"])

    @classmethod
    def from_csv(cls, path: str | Path) -> FixtureCalendar:
        return cls(pd.read_csv(path))

    def games_between(self, team: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Non-league games played in [start, end)."""
        dates = self.dates.get(team, [])
        return bisect.bisect_left(dates, end) - bisect.bisect_left(dates, start)

    def last_before(self, team: str, when: pd.Timestamp) -> pd.Timestamp | None:
        dates = self.dates.get(team, [])
        i = bisect.bisect_left(dates, when)
        return dates[i - 1] if i > 0 else None


def load_manager_changes(path_or_df: str | Path | pd.DataFrame | None) -> dict[str, list[pd.Timestamp]]:
    """Reads manager appointments (columns Team, Date) into team -> sorted appointment dates."""
    if path_or_df is None:
        return {}
    df = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df
    if df.empty:
        return {}
    df = df.assign(Date=pd.to_datetime(df["Date"]), Team=df["Team"].map(normalize_team_name))
    return {t: sorted(g["Date"]) for t, g in df.groupby("Team")}


def latest_change_before(changes: dict[str, list[pd.Timestamp]], when: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Most recent appointment per team on or before `when` (for down-weighting older matches)."""
    out = {}
    for team, dates in changes.items():
        before = [d for d in dates if d <= when]
        if before:
            out[team] = before[-1]
    return out


def assign_seasons(dates: pd.Series) -> np.ndarray:
    """Season index per match, starting a new season after a long break."""
    d = pd.to_datetime(dates).sort_values()
    new = d.diff().dt.days.fillna(0) > SEASON_GAP_DAYS
    return new.cumsum().reindex(pd.to_datetime(dates).index).to_numpy()
