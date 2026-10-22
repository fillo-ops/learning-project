import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from api import main as api_main


@pytest.fixture()
def client(trained_engine, monkeypatch):
    monkeypatch.setattr(api_main, "get_engine", lambda: trained_engine)
    return TestClient(api_main.app)


def test_predict_endpoint(client):
    resp = client.get(
        "/predict", params={"home": "Arsenal", "away": "Chelsea", "b365h": 2.0, "b365d": 3.4, "b365a": 3.8}
    )
    assert resp.status_code == 200
    assert resp.json()["value_bets"]


def test_unknown_team_is_422(client):
    assert client.get("/predict", params={"home": "Nowhere", "away": "Chelsea"}).status_code == 422


def test_partial_odds_is_422(client):
    assert client.get("/predict", params={"home": "Arsenal", "away": "Chelsea", "b365h": 2.0}).status_code == 422
