"""FastAPI prediction service: uvicorn api.main:app --reload"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from src.config import DEFAULT_MODEL_PATH
from src.model import MatchPredictorEngine
from src.pipeline import load_or_train

app = FastAPI(title="Football Match Predictor API", version="2.1.0")


@lru_cache(maxsize=1)
def get_engine() -> MatchPredictorEngine:
    """Loads (or trains once) the model and keeps it in memory across requests."""
    return load_or_train(DEFAULT_MODEL_PATH)


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
