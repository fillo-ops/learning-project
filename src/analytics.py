"""One entry point for team-unit analysis: ratings, tactics, players and simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.context import latest_change_before
from src.game_state import DisciplineModel
from src.names import normalize_team_name
from src.players import PlayerRatings
from src.simulation import MatchSimulator
from src.tactics import tactical_matchup_notes, tactical_profiles
from src.team_ratings import TeamUnitRatings, season_timeline


class TeamAnalytics:
    """Rates each team's attack, midfield and defence, profiles tactics, and simulates matchups."""

    def __init__(
        self,
        matches: pd.DataFrame,
        players: pd.DataFrame | None = None,
        manager_changes: dict[str, list[pd.Timestamp]] | None = None,
        **simulator_kwargs: Any,
    ):
        self.matches = matches
        latest = pd.to_datetime(matches["Date"]).max()
        current = latest_change_before(manager_changes or {}, latest)
        self.manager_changes = current
        self.ratings = TeamUnitRatings().fit(matches, manager_changes=current)
        self.discipline = DisciplineModel().fit(matches)
        # A new manager's style replaces the old one: profile only the matches since the appointment
        self.profiles = tactical_profiles(matches, since=current)
        simulator_kwargs.setdefault("discipline", self.discipline)
        self.players = PlayerRatings(players) if players is not None else None
        self.simulator = MatchSimulator(self.ratings, self.players, **simulator_kwargs)

    @classmethod
    def from_csv(cls, matches_csv: str | Path, players_csv: str | Path | None = None, **kwargs: Any) -> TeamAnalytics:
        from src.data_loader import DataLoader

        matches = DataLoader().load_csv_safely(matches_csv)
        players = pd.read_csv(players_csv) if players_csv and Path(players_csv).exists() else None
        return cls(matches, players, **kwargs)

    @property
    def teams(self) -> list[str]:
        return list(self.ratings.teams_)

    def team_report(self, team: str, timeline_days: int = 365) -> dict[str, Any]:
        """Unit indices with league ranks, tactical style, and how the units evolved over the season."""
        t = normalize_team_name(team)
        profile = self.ratings.unit_profile(t)  # raises ValueError for unknown teams
        table = self.ratings.league_table()
        ranks = {unit: int(table[unit].rank(ascending=False, method="min")[t]) for unit in profile}
        start = pd.to_datetime(self.matches["Date"]).max() - pd.Timedelta(days=timeline_days)
        timeline = season_timeline(self.matches, teams=[t], start=start)
        report: dict[str, Any] = {
            "team": t,
            "units": profile,
            "league_rank": ranks,
            "n_teams": len(table),
            "style": self.profiles.at[t, "style"],
            "style_metrics": self.profiles.loc[t].drop("style").to_dict(),
            "timeline": timeline.drop(columns="team")
            .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records"),
        }
        if self.players is not None:
            report["player_units"] = self.players.team_units(t)
            report["players"] = self.players.player_table(t).head(15).to_dict(orient="records")
        return report

    def simulate(self, home: str, away: str, **kwargs: Any) -> dict[str, Any]:
        result = self.simulator.simulate(home, away, **kwargs)
        result["tactical_notes"] = tactical_matchup_notes(self.profiles, result["home_team"], result["away_team"])
        result["home"]["style"] = self.profiles.at[result["home_team"], "style"]
        result["away"]["style"] = self.profiles.at[result["away_team"], "style"]
        return result
