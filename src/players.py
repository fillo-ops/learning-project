"""Individual player ratings and their effect on team units.

Expects a player-season CSV (e.g. exported from FBref) with columns:

    player, team, position (GK/DEF/MID/FWD), minutes,
    goals, assists, shots, shots_on_target, key_passes, tackles, interceptions

Missing stat columns are treated as zero. Ratings are per-90 composites z-scored within
each position group (50 = average for the position). Absence effects are transparent
assumptions configured in PLAYER_CONFIG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PLAYER_CONFIG
from src.names import normalize_team_name

STAT_COLS = ["goals", "assists", "shots", "shots_on_target", "key_passes", "tackles", "interceptions"]
POSITION_ALIASES = {"G": "GK", "D": "DEF", "DF": "DEF", "M": "MID", "MF": "MID", "F": "FWD", "FW": "FWD", "ST": "FWD"}

# unit -> per-90 stats (and weights) in its composite
UNIT_COMPOSITES = {
    "attack": {"goals": 1.0, "shots": 1.0, "shots_on_target": 1.0, "assists": 0.5, "key_passes": 0.5},
    "midfield": {"key_passes": 1.0, "assists": 0.5, "tackles": 1.0, "interceptions": 1.0},
    "defence": {"tackles": 1.0, "interceptions": 1.0},
}
# which positions make up each team unit
UNIT_POSITIONS = {"attack": ("FWD", "MID"), "midfield": ("MID",), "defence": ("DEF",)}

NEUTRAL_EFFECTS = {
    "shots_for": 1.0,
    "finishing": 1.0,
    "shots_against": 1.0,
    "on_target_against": 1.0,
    "conversion_against": 1.0,
}


class PlayerRatings:
    """Rates players by unit and estimates how absences change a team's attack, midfield and defence."""

    def __init__(self, players: pd.DataFrame, config: dict | None = None):
        self.config = {**PLAYER_CONFIG, **(config or {})}
        self.players = self._prepare(players)

    @classmethod
    def from_csv(cls, path: str | Path, config: dict | None = None) -> PlayerRatings:
        return cls(pd.read_csv(path), config)

    def _prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns=str.lower).copy()
        missing = {"player", "team", "position", "minutes"} - set(df.columns)
        if missing:
            raise ValueError(f"Player data missing required columns: {sorted(missing)}")
        df["team"] = df["team"].map(normalize_team_name)
        pos = df["position"].astype(str).str.upper().str.split(r"[,/ ]").str[0]
        df["position"] = pos.map(lambda p: POSITION_ALIASES.get(p, p))
        bad = ~df["position"].isin(["GK", "DEF", "MID", "FWD"])
        if bad.any():
            raise ValueError(f"Unrecognised positions: {sorted(df.loc[bad, 'position'].unique())}")
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0).clip(lower=0)
        for col in STAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df else 0.0
            df[f"{col}_p90"] = np.where(df["minutes"] > 0, df[col] * 90.0 / df["minutes"].clip(lower=1), 0.0)

        rated = df["minutes"] >= self.config["min_minutes"]
        df["small_sample"] = ~rated
        for unit, weights in UNIT_COMPOSITES.items():
            composite = pd.Series(0.0, index=df.index)
            for stat, w in weights.items():
                col = f"{stat}_p90"
                grp = df[rated].groupby("position")[col]
                mean = df["position"].map(grp.mean())
                sd = df["position"].map(grp.std(ddof=0)).replace(0, np.nan)
                composite += w * ((df[col] - mean) / sd).fillna(0.0)
            ref = composite[rated].groupby(df.loc[rated, "position"])
            z = (composite - df["position"].map(ref.mean())) / df["position"].map(ref.std(ddof=0)).replace(0, np.nan)
            df[f"{unit}_rating"] = (50.0 + 15.0 * z.fillna(0.0)).clip(1, 99).round(1)

        # Share of the team's season output each player accounts for
        team_tot = df.groupby("team")[STAT_COLS].transform("sum").replace(0, np.nan)
        df["shot_share"] = (df["shots"] / team_tot["shots"]).fillna(0.0)
        df["sot_share"] = (df["shots_on_target"] / team_tot["shots_on_target"]).fillna(0.0)
        df["goal_share"] = (df["goals"] / team_tot["goals"]).fillna(0.0)
        mid_work = df["key_passes"] + df["tackles"] + df["interceptions"]
        df["midfield_share"] = (mid_work / mid_work.groupby(df["team"]).transform("sum").replace(0, np.nan)).fillna(0.0)
        def_work = df["tackles"] + df["interceptions"]
        df["defence_share"] = (def_work / def_work.groupby(df["team"]).transform("sum").replace(0, np.nan)).fillna(0.0)
        # Rough availability: fraction of the busiest team-mate's minutes
        df["availability"] = (df["minutes"] / df.groupby("team")["minutes"].transform("max").replace(0, np.nan)).fillna(
            0.0
        )
        return df

    def _team(self, team: str) -> pd.DataFrame:
        t = normalize_team_name(team)
        squad = self.players[self.players["team"] == t]
        if squad.empty:
            raise ValueError(f"No player data for team {team!r}")
        return squad

    def _find(self, team: str, names: list[str] | None) -> pd.DataFrame:
        squad = self._team(team)
        if not names:
            return squad.iloc[0:0]
        wanted = {n.casefold().strip() for n in names}
        found = squad[squad["player"].str.casefold().str.strip().isin(wanted)]
        unknown = wanted - set(found["player"].str.casefold().str.strip())
        if unknown:
            raise ValueError(f"Players not found in {team}: {sorted(unknown)}")
        return found

    def player_table(self, team: str | None = None) -> pd.DataFrame:
        df = self._team(team) if team else self.players
        cols = ["player", "team", "position", "minutes", "goals", "assists", "attack_rating", "midfield_rating",
                "defence_rating", "shot_share", "goal_share", "small_sample"]  # fmt: skip
        return df[cols].sort_values(["team", "minutes"], ascending=[True, False]).reset_index(drop=True)

    def team_units(self, team: str, absent: list[str] | None = None) -> dict[str, float]:
        """Minutes-weighted player ratings for each unit (50 = average), excluding absent players."""
        squad = self._team(team)
        out_ids = set(self._find(team, absent).index)
        available = squad[~squad.index.isin(out_ids) & ~squad["small_sample"]]
        units = {}
        for unit, positions in UNIT_POSITIONS.items():
            grp = available[available["position"].isin(positions)]
            w = grp["minutes"]
            units[unit] = round(float(np.average(grp[f"{unit}_rating"], weights=w)), 1) if w.sum() > 0 else float("nan")
        return units

    def absence_effects(self, team: str, absent: list[str] | None) -> dict[str, float]:
        """Multipliers on the team's simulation rates when the listed players are unavailable."""
        effects = dict(NEUTRAL_EFFECTS)
        cfg = self.config
        loss = 1.0 - cfg["replacement_level"]
        for _, p in self._find(team, absent).iterrows():
            a = float(p["availability"])  # a rotation player was often absent anyway
            effects["shots_for"] *= 1.0 - a * loss * p["shot_share"]
            # Losing a player who converts better than his shot share implies hurts finishing
            effects["finishing"] *= 1.0 - a * loss * float(np.clip(p["goal_share"] - p["sot_share"], -0.2, 0.2))
            if p["position"] == "MID":
                m = a * loss * cfg["midfield_weight"] * p["midfield_share"]
                effects["shots_for"] *= 1.0 - m
                effects["shots_against"] *= 1.0 + m
            elif p["position"] == "DEF":
                effects["on_target_against"] *= 1.0 + a * loss * cfg["defence_weight"] * p["defence_share"]
            elif p["position"] == "GK":
                effects["conversion_against"] *= 1.0 + a * cfg["keeper_weight"]
        return {k: round(float(v), 4) for k, v in effects.items()}

    def scorer_weights(self, team: str, absent: list[str] | None = None) -> pd.Series:
        """Probability that a given team goal is scored by each available player."""
        squad = self._team(team)
        out_ids = set(self._find(team, absent).index)
        avail = squad[~squad.index.isin(out_ids)]
        threat = 0.5 * avail["goals_p90"] / max(avail["goals_p90"].sum(), 1e-9) + 0.5 * avail["shots_p90"] / max(
            avail["shots_p90"].sum(), 1e-9
        )
        w = threat * avail["availability"]
        if w.sum() <= 0:
            w = avail["availability"]
        return (w / w.sum()).set_axis(avail["player"])
