"""Tactical style profiles from match statistics.

Football-Data.co.uk CSVs carry no possession or event data, so styles are inferred from
proxies: shot volume for/against (territory), shot accuracy (chance quality), conversion,
corners (wide play / set pieces) and fouls (physicality). Labels are descriptive heuristics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.names import normalize_team_name

# metric -> (home column, away column); "for" metrics read the team's own column
_STAT_PAIRS = {
    "goals": ("FTHG", "FTAG"),
    "shots": ("HS", "AS"),
    "sot": ("HST", "AST"),
    "corners": ("HC", "AC"),
    "fouls": ("HF", "AF"),
}


def team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match with stats for and against (missing columns become NaN)."""
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    frames = []
    for venue, (own, opp) in {"home": ("HomeTeam", "AwayTeam"), "away": ("AwayTeam", "HomeTeam")}.items():
        out = pd.DataFrame({"Date": data["Date"], "team": data[own].map(normalize_team_name), "venue": venue})
        out["opponent"] = data[opp].map(normalize_team_name)
        for metric, (hcol, acol) in _STAT_PAIRS.items():
            mine, theirs = (hcol, acol) if venue == "home" else (acol, hcol)
            out[f"{metric}_for"] = pd.to_numeric(data.get(mine), errors="coerce") if mine in data else np.nan
            out[f"{metric}_against"] = pd.to_numeric(data.get(theirs), errors="coerce") if theirs in data else np.nan
        frames.append(out)
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "team"], kind="stable").reset_index(drop=True)


def style_metrics(df: pd.DataFrame, last_n: int | None = 19) -> pd.DataFrame:
    """Per-team style metrics over each team's most recent `last_n` matches (None = all)."""
    log = team_match_log(df)
    if last_n:
        log = log.groupby("team", group_keys=False).tail(last_n)
    g = log.groupby("team")
    totals = g.sum(numeric_only=True)
    n = g.size()
    out = pd.DataFrame(index=totals.index)
    out["matches"] = n
    out["shots_pg"] = totals["shots_for"] / n
    out["shots_conceded_pg"] = totals["shots_against"] / n
    out["shot_share"] = totals["shots_for"] / (totals["shots_for"] + totals["shots_against"])
    out["accuracy"] = totals["sot_for"] / totals["shots_for"]
    out["conversion"] = totals["goals_for"] / totals["sot_for"].replace(0, np.nan)
    out["accuracy_allowed"] = totals["sot_against"] / totals["shots_against"]
    out["conversion_allowed"] = totals["goals_against"] / totals["sot_against"].replace(0, np.nan)
    out["goals_pg"] = totals["goals_for"] / n
    out["goals_conceded_pg"] = totals["goals_against"] / n
    if log["corners_for"].notna().any():
        out["corners_pg"] = g["corners_for"].mean()
    if log["fouls_for"].notna().any():
        out["fouls_pg"] = g["fouls_for"].mean()
    return out.round(3)


_TRAITS = [
    # (metric, direction, z threshold, label)
    ("shot_share", +1, 0.8, "Dominant: controls territory and shot volume"),
    ("shot_share", -1, 0.8, "Reactive: cedes territory"),
    ("accuracy", +1, 0.8, "Creates high-quality chances"),
    ("conversion", +1, 0.8, "Clinical finishing"),
    ("conversion", -1, 0.8, "Wasteful finishing"),
    ("shots_conceded_pg", -1, 0.8, "Suppresses opponent shots (press / control)"),
    ("accuracy_allowed", -1, 0.8, "Compact block: forces low-quality shots"),
    ("accuracy_allowed", +1, 0.8, "Leaky: concedes clear chances"),
    ("corners_pg", +1, 0.8, "Wide play / set-piece threat"),
    ("fouls_pg", +1, 0.8, "Aggressive, physical"),
    ("fouls_pg", -1, 0.8, "Low-contact, technical"),
]


def tactical_profiles(df: pd.DataFrame, last_n: int | None = 19) -> pd.DataFrame:
    """Style metrics plus league z-scores and descriptive tactical labels for every team."""
    metrics = style_metrics(df, last_n=last_n)
    z = (metrics - metrics.mean()) / metrics.std(ddof=0).replace(0, np.nan)
    labels = []
    for team in metrics.index:
        traits = [lab for col, sign, thr, lab in _TRAITS if col in z and sign * z.at[team, col] >= thr]
        if z.at[team, "shot_share"] <= -0.5 and z.at[team, "conversion"] >= 0.5:
            traits.append("Counter-attacking: few shots but punishing")
        labels.append("; ".join(traits) if traits else "Balanced")
    metrics["style"] = labels
    return metrics


def tactical_matchup_notes(profiles: pd.DataFrame, home: str, away: str) -> list[str]:
    """Plain-language notes on where the two styles collide."""
    h, a = normalize_team_name(home), normalize_team_name(away)
    p = profiles
    notes = []
    share_gap = p.at[h, "shot_share"] - p.at[a, "shot_share"]
    if abs(share_gap) >= 0.05:
        leader = h if share_gap > 0 else a
        notes.append(f"{leader} should control territory (shot share gap {abs(share_gap):.0%}).")
    for att, dfn in ((h, a), (a, h)):
        if (
            p.at[att, "accuracy"] > p["accuracy"].median()
            and p.at[dfn, "accuracy_allowed"] < p["accuracy_allowed"].median()
        ):
            notes.append(f"{att}'s chance quality meets {dfn}'s compact block: expect fewer clear-cut chances.")
        if p.at[att, "conversion"] > p["conversion"].quantile(0.75):
            notes.append(f"{att} finish clinically, so even limited possession is dangerous for {dfn}.")
    if "fouls_pg" in p and max(p.at[h, "fouls_pg"], p.at[a, "fouls_pg"]) > p["fouls_pg"].quantile(0.8):
        notes.append("High foul rate expected: set pieces and cards may shape the game.")
    return notes
