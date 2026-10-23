"""Pydantic response models: typed OpenAPI schemas for /docs and generated clients.

Core fields are typed; deeply nested analytical detail is kept as `extra="allow"` models or
dicts so new diagnostics can be added without breaking clients.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Probabilities(BaseModel):
    home_win: float
    draw: float
    away_win: float


class Scoreline(BaseModel):
    score: str
    probability: float


class ValueBet(BaseModel):
    ev: float
    kelly_stake: float
    is_value: bool


class AsianLine(BaseModel):
    line: float
    win: float
    half_win: float
    push: float
    half_loss: float
    loss: float
    fair_odds: float | None = Field(None, description="Zero-EV decimal odds for the home side / over")
    under_fair_odds: float | None = Field(None, description="Goal lines only: zero-EV odds for the under")


class AsianLines(BaseModel):
    asian_handicap_home: list[AsianLine]
    goal_lines_over: list[AsianLine]


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    probabilities: Probabilities
    component_probabilities: dict[str, Any]
    expected_goals: dict[str, float]
    elo_ratings: dict[str, float]
    markets: dict[str, float]
    top_scorelines: list[Scoreline]
    asian_lines: AsianLines
    value_bets: dict[str, ValueBet]


class UnitRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    team: str
    attack: float
    defence: float
    midfield: float
    overall: float


class UnitRatingResponse(BaseModel):
    units: list[UnitRow]


class GameStateSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    p_goalless: float
    home_scores_first: float
    away_scores_first: float
    home_win_if_scores_first: float | None
    away_win_if_scores_first: float | None
    home_comeback_win: float
    away_comeback_win: float
    p_home_red_card: float
    p_away_red_card: float
    goals_by_period: dict[str, dict[str, float]]


class SimulationMarkets(AsianLines):
    over_2_5: float
    btts: float


class SimulationSide(BaseModel):
    model_config = ConfigDict(extra="allow")
    team: str
    attack: dict[str, Any]
    defence: dict[str, Any]
    midfield_effect: dict[str, Any]
    style: str | None = None


class SimulationResponse(BaseModel):
    home_team: str
    away_team: str
    n_sims: int
    midfield_shift: float
    possession_shift: float
    volume_stat: str
    expected_possession: dict[str, float] | None
    outcome: Probabilities
    top_scorelines: list[Scoreline]
    unit_comparison: list[dict[str, Any]]
    markets: SimulationMarkets
    game_state: GameStateSummary | None
    home: SimulationSide
    away: SimulationSide
    tactical_notes: list[str]
    midfield_sensitivity: list[dict[str, Any]]


class BetIn(BaseModel):
    match: str
    selection: str
    prob: float = Field(gt=0, lt=1)
    odds: float = Field(gt=1)


class PortfolioRequest(BaseModel):
    bets: list[BetIn]
    max_total_exposure: float = Field(0.12, gt=0, le=1)


class PortfolioBet(BetIn):
    ev: float
    kelly_stake: float
    stake: float
    scaled_by: float


class PortfolioResponse(BaseModel):
    bets: list[PortfolioBet]
    total_exposure: float
