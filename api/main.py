"""FastAPI prediction service: uvicorn api.main:app --reload  (interactive docs at /docs)"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query

from api.schemas import (
    AsianLines,
    PortfolioRequest,
    PortfolioResponse,
    PredictionResponse,
    SimulationResponse,
    UnitRatingResponse,
)
from src.analytics import TeamAnalytics
from src.config import DEFAULT_MODEL_PATH
from src.market import asian_markets, line_movement, portfolio_kelly
from src.model import MatchPredictorEngine
from src.pipeline import load_analytics, load_or_train

app = FastAPI(title="Football Match Predictor API", version="2.3.0")


@lru_cache(maxsize=1)
def get_engine() -> MatchPredictorEngine:
    """Loads (or trains once) the model and keeps it in memory across requests."""
    return load_or_train(DEFAULT_MODEL_PATH)


@lru_cache(maxsize=1)
def get_analytics() -> TeamAnalytics:
    """Team unit ratings, tactics, players and simulator, built once per process."""
    return load_analytics()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/teams")
def teams(competition: str | None = Query(None, description='e.g. "Champions League"')) -> dict:
    if competition is None:
        return {"teams": get_engine().teams}
    try:
        return {"competition": competition, "teams": get_analytics().teams_in(competition)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/competitions")
def competitions() -> dict:
    return {"competitions": get_analytics().competitions}


@app.get("/predict", response_model=PredictionResponse)
def predict(
    home: str = Query(..., description="Home team name"),
    away: str = Query(..., description="Away team name"),
    b365h: float | None = Query(None, gt=1.0),
    b365d: float | None = Query(None, gt=1.0),
    b365a: float | None = Query(None, gt=1.0),
    date: str | None = Query(None, description="Match date YYYY-MM-DD"),
) -> dict:
    prices = (b365h, b365d, b365a)
    if any(p is not None for p in prices) and not all(p is not None for p in prices):
        raise HTTPException(status_code=422, detail="Provide all three odds (b365h, b365d, b365a) or none")
    odds = prices if all(p is not None for p in prices) else None
    try:
        return get_engine().predict_fixture(home, away, odds=odds, match_date=date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/units", response_model=UnitRatingResponse)
def league_units(
    competition: str | None = Query(None, description='Only clubs in this competition, e.g. "Champions League"'),
) -> dict:
    """Attack / midfield / defence indices (100 = average of all clubs) for every team."""
    try:
        table = get_analytics().league_table(competition)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"units": table.reset_index(names="team").to_dict(orient="records")}


@app.get("/teams/{team}/report")
def team_report(team: str) -> dict:
    """Season unit report: indices, league ranks, tactical style, timeline and players."""
    try:
        return get_analytics().team_report(team)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/teams/{team}/attack-vs-defences")
def attack_vs_defences(team: str, at_home: bool = True) -> dict:
    try:
        table = get_analytics().simulator.attack_vs_all_defences(team, at_home=at_home)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"team": team, "at_home": at_home, "opponents": table.to_dict(orient="records")}


@lru_cache(maxsize=256)
def _cached_simulation(
    analytics_id: int,
    home: str,
    away: str,
    midfield_shift: float,
    possession_shift: float,
    home_absent: tuple[str, ...],
    away_absent: tuple[str, ...],
    referee: str | None,
) -> dict[str, Any]:
    """20,000-run simulations are deterministic (seeded), so identical what-ifs are served from memory."""
    analytics = get_analytics()
    result = analytics.simulate(
        home,
        away,
        midfield_shift=midfield_shift,
        possession_shift=possession_shift,
        home_absent=list(home_absent) or None,
        away_absent=list(away_absent) or None,
        referee=referee,
    )
    result["midfield_sensitivity"] = analytics.simulator.midfield_sensitivity(home, away).to_dict(orient="records")
    return result


@app.get("/simulate", response_model=SimulationResponse)
def simulate(
    home: str = Query(..., description="Home team name"),
    away: str = Query(..., description="Away team name"),
    midfield_shift: float = Query(0.0, ge=-1.0, le=1.0, description="Tilt the midfield battle (+ favours home)"),
    possession_shift: float = Query(0.0, ge=-40.0, le=40.0, description="Extra possession points for the home side"),
    home_absent: Annotated[list[str] | None, Query(description="Unavailable home players")] = None,
    away_absent: Annotated[list[str] | None, Query(description="Unavailable away players")] = None,
    referee: str | None = Query(None, description="Referee (scales red-card risk)"),
) -> dict:
    """Monte Carlo game-state simulation: goals each attack can score, how each defence fares, midfield impact."""
    analytics = get_analytics()
    try:
        result = _cached_simulation(
            id(analytics),
            home,
            away,
            midfield_shift,
            possession_shift,
            tuple(sorted(home_absent or [])),
            tuple(sorted(away_absent or [])),
            referee,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return copy.deepcopy(result)


@app.get("/markets/asian", response_model=AsianLines)
def asian_lines(home: str = Query(...), away: str = Query(...)) -> dict:
    """Fair Asian handicap and goal-line odds from the simulated joint score distribution."""
    try:
        matrix = get_analytics().simulator.score_matrix(home, away)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return asian_markets(matrix)


@app.get("/markets/line-movement")
def market_line_movement() -> dict:
    """Opening-to-closing price moves per team and whether moves carried information."""
    try:
        table, summary = line_movement(get_analytics().matches)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"summary": summary, "teams": table.reset_index().to_dict(orient="records")}


@app.post("/portfolio", response_model=PortfolioResponse)
def portfolio(request: PortfolioRequest) -> dict:
    """Matchday staking: one bet per match, fractional Kelly, total exposure capped."""
    picks = portfolio_kelly([b.model_dump() for b in request.bets], max_total_exposure=request.max_total_exposure)
    return {"bets": picks, "total_exposure": round(sum(p["stake"] for p in picks), 4)}
