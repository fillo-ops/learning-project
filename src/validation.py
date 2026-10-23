"""Schema and sanity checks for match data (catches renamed headers and corrupted rows)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data_loader import parse_match_dates

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
EXPECTED_FOOTBALL_DATA = ["HS", "AS", "HST", "AST", "B365H", "B365D", "B365A"]
ODDS_COLS = ["B365H", "B365D", "B365A", "PSCH", "PSCD", "PSCA", "AvgH", "AvgD", "AvgA"]


@dataclass
class ValidationReport:
    rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = [f"**Rows checked:** {self.rows}", ""]
        lines += ["### Errors", *(f"- {e}" for e in self.errors)] if self.errors else ["No errors."]
        if self.warnings:
            lines += ["", "### Warnings", *(f"- {w}" for w in self.warnings)]
        return "\n".join(lines)


def validate_matches(df: pd.DataFrame, expect_football_data: bool = True) -> ValidationReport:
    """Checks columns, types and value ranges; errors fail the check, warnings are informational."""
    report = ValidationReport(rows=len(df))
    if df.empty:
        report.errors.append("File has no rows")
        return report

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required columns (renamed header?): {missing}")
        return report
    if expect_football_data:
        absent = [c for c in EXPECTED_FOOTBALL_DATA if c not in df.columns]
        if absent:
            report.warnings.append(f"Usual Football-Data columns absent: {absent}")

    complete = df.dropna(subset=REQUIRED)
    if len(complete) < len(df):
        report.warnings.append(f"{len(df) - len(complete)} rows have blanks in required columns (unplayed fixtures?)")

    dates = parse_match_dates(complete["Date"])
    if dates.isna().any():
        report.errors.append(
            f"{int(dates.isna().sum())} unparseable dates, e.g. {complete.loc[dates.isna(), 'Date'].iloc[0]!r}"
        )

    for col in ("FTHG", "FTAG"):
        goals = pd.to_numeric(complete[col], errors="coerce")
        bad = goals.isna() | (goals < 0) | (goals > 15) | (goals % 1 != 0)
        if bad.any():
            report.errors.append(f"{col}: {int(bad.sum())} non-integer or out-of-range values")

    ftr = complete["FTR"].astype(str).str.strip()
    if (~ftr.isin(["H", "D", "A"])).any():
        report.errors.append(f"FTR has unexpected values: {sorted(set(ftr) - {'H', 'D', 'A'})[:5]}")
    hg = pd.to_numeric(complete["FTHG"], errors="coerce")
    ag = pd.to_numeric(complete["FTAG"], errors="coerce")
    implied = (hg > ag).map({True: "H"}).fillna((hg == ag).map({True: "D"})).fillna("A")
    mismatch = (implied != ftr) & hg.notna() & ag.notna()
    if mismatch.any():
        report.errors.append(f"{int(mismatch.sum())} rows where FTR disagrees with the score")

    for shots, on_target in (("HS", "HST"), ("AS", "AST")):
        if {shots, on_target} <= set(complete.columns):
            s, t = pd.to_numeric(complete[shots], errors="coerce"), pd.to_numeric(complete[on_target], errors="coerce")
            bad = (t > s) & s.notna() & t.notna()
            if bad.any():
                report.errors.append(f"{int(bad.sum())} rows with more {on_target} than {shots}")

    for col in [c for c in ODDS_COLS if c in complete.columns]:
        odds = pd.to_numeric(complete[col], errors="coerce")
        if (odds <= 1.0).any():
            report.errors.append(f"{col}: {int((odds <= 1.0).sum())} prices at or below 1.0")
        if odds.isna().mean() > 0.2:
            report.warnings.append(f"{col}: {odds.isna().mean():.0%} missing")

    if {"HPoss", "APoss"} <= set(complete.columns):
        total = pd.to_numeric(complete["HPoss"], errors="coerce") + pd.to_numeric(complete["APoss"], errors="coerce")
        scale = 100.0 if total.median() > 2 else 1.0
        bad = (total - scale).abs() > 0.02 * scale
        if bad.any():
            report.errors.append(f"{int(bad.sum())} rows where possession does not sum to {scale:g}")

    dupes = complete.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum()
    if dupes:
        report.errors.append(f"{int(dupes)} duplicated fixtures")

    teams = set(complete["HomeTeam"]).union(complete["AwayTeam"])
    if len(teams) % 2 or not 10 <= len(teams) <= 24:
        report.warnings.append(f"Unusual number of teams: {len(teams)}")
    return report
