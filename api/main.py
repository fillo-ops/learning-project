"""FastAPI prediction service: uvicorn api.main:app --reload"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from src.analytics import TeamAnalytics
from src.config import DEFAULT_MODEL_PATH
from src.model import MatchPredictorEngine
from src.pipeline import load_analytics, load_or_train

app = FastAPI(title="Football Match Predictor API", version="2.2.0")


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
def teams() -> dict:
    return {"teams": get_engine().teams}


@app.get("/predict")
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


@app.get("/units")
def league_units() -> dict:
    """Attack / midfield / defence indices (100 = league average) for every team."""
    table = get_analytics().ratings.league_table()
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


@app.get("/simulate")
def simulate(
    home: str = Query(..., description="Home team name"),
    away: str = Query(..., description="Away team name"),
    midfield_shift: float = Query(0.0, ge=-1.0, le=1.0, description="Tilt the midfield battle (+ favours home)"),
    home_absent: Annotated[list[str] | None, Query(description="Unavailable home players")] = None,
    away_absent: Annotated[list[str] | None, Query(description="Unavailable away players")] = None,
) -> dict:
    """Monte Carlo simulation of a fixture: goals each attack can score, how each defence fares, midfield impact."""
    analytics = get_analytics()
    try:
        result = analytics.simulate(
            home, away, midfield_shift=midfield_shift, home_absent=home_absent, away_absent=away_absent
        )
        result["midfield_sensitivity"] = analytics.simulator.midfield_sensitivity(home, away).to_dict(orient="records")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result
