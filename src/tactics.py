"""Tactical style profiles from match statistics.

Uses possession share and chances created when the data has them (HPoss/APoss, HCC/ACC),
plus shot volume, shots on target, conversion, corners and fouls. Labels are descriptive
heuristics based on league z-scores.
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
    "chances": ("HCC", "ACC"),
    "possession": ("HPoss", "APoss"),
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
    out["sot_pg"] = totals["sot_for"] / n
    out["sot_conceded_pg"] = totals["sot_against"] / n
    out["accuracy"] = totals["sot_for"] / totals["shots_for"]
    out["conversion"] = totals["goals_for"] / totals["sot_for"].replace(0, np.nan)
    out["accuracy_allowed"] = totals["sot_against"] / totals["shots_against"]
    out["conversion_allowed"] = totals["goals_against"] / totals["sot_against"].replace(0, np.nan)
    out["goals_pg"] = totals["goals_for"] / n
    out["goals_conceded_pg"] = totals["goals_against"] / n
    if log["possession_for"].notna().any():
        poss = log["possession_for"].where(log["possession_for"] > 1.0, log["possession_for"] * 100.0)
        out["possession_pct"] = poss.groupby(log["team"]).mean()
    if log["chances_for"].notna().any():
        out["chances_pg"] = g["chances_for"].mean()
        out["chances_conceded_pg"] = g["chances_against"].mean()
        out["chance_share"] = g["chances_for"].sum() / (g["chances_for"].sum() + g["chances_against"].sum())
    if log["corners_for"].notna().any():
        out["corners_pg"] = g["corners_for"].mean()
    if log["fouls_for"].notna().any():
        out["fouls_pg"] = g["fouls_for"].mean()
    return out.round(3)


MAX_TRAITS = 4

_TRAITS = [
    # (metric, direction, z threshold, label)
    ("possession_pct", +1, 0.8, "Possession-based: keeps the ball"),
    ("possession_pct", -1, 0.8, "Direct: happy without the ball"),
    ("chances_pg", +1, 0.8, "Creative: high chance creation"),
    ("chances_conceded_pg", -1, 0.8, "Restricts opponent chances"),
    ("sot_pg", +1, 0.8, "Tests the keeper often (high shots on target)"),
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
        scored = [
            (sign * z.at[team, col], lab)
            for col, sign, thr, lab in _TRAITS
            if col in z and sign * z.at[team, col] >= thr
        ]
        # Keep the most pronounced traits so the label stays readable
        traits = [lab for _, lab in sorted(scored, reverse=True)[:MAX_TRAITS]]
        control = "possession_pct" if "possession_pct" in z else "shot_share"
        if z.at[team, control] <= -0.5 and z.at[team, "conversion"] >= 0.5:
            traits.insert(0, "Counter-attacking: few shots but punishing")
        labels.append("; ".join(traits) if traits else "Balanced")
    metrics["style"] = labels
    return metrics


def tactical_matchup_notes(profiles: pd.DataFrame, home: str, away: str) -> list[str]:
    """Plain-language notes on where the two styles collide."""
    h, a = normalize_team_name(home), normalize_team_name(away)
    p = profiles
    notes = []
    if "possession_pct" in p:
        gap = p.at[h, "possession_pct"] - p.at[a, "possession_pct"]
        if abs(gap) >= 5:
            leader, other = (h, a) if gap > 0 else (a, h)
            notes.append(
                f"{leader} average {p.at[leader, 'possession_pct']:.0f}% possession vs {other}'s "
                f"{p.at[other, 'possession_pct']:.0f}%: expect {leader} to dictate the ball."
            )
    if "chance_share" in p:
        gap = p.at[h, "chance_share"] - p.at[a, "chance_share"]
        if abs(gap) >= 0.05:
            leader = h if gap > 0 else a
            notes.append(f"{leader} win the chance-creation battle more often (chance share gap {abs(gap):.0%}).")
    else:
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
