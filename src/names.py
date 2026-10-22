"""Canonical team name normalization across CSVs, bookmakers and APIs."""

from __future__ import annotations

TEAM_ALIASES: dict[str, str] = {
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "West Ham": "West Ham United",
    "Brighton & Hove Albion": "Brighton",
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Ipswich": "Ipswich Town",
    "Luton": "Luton Town",
}

_ALIASES_CASEFOLD = {k.casefold(): v for k, v in TEAM_ALIASES.items()}


def normalize_team_name(name: object) -> str:
    """Maps a raw team name onto its canonical spelling (case-insensitive, whitespace-tolerant)."""
    cleaned = " ".join(str(name).split())
    return _ALIASES_CASEFOLD.get(cleaned.casefold(), cleaned)
