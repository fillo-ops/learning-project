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


@pytest.fixture()
def analytics_client(analytics, monkeypatch):
    monkeypatch.setattr(api_main, "get_analytics", lambda: analytics)
    return TestClient(api_main.app)


def test_units_and_report_endpoints(analytics_client):
    assert len(analytics_client.get("/units").json()["units"]) == 20
    assert analytics_client.get("/teams/Arsenal/report").json()["team"] == "Arsenal"
    assert analytics_client.get("/teams/Atlantis/report").status_code == 404
    assert analytics_client.get("/teams/Arsenal/attack-vs-defences").status_code == 200


def test_simulate_endpoint(analytics_client):
    resp = analytics_client.get(
        "/simulate",
        params={"home": "Arsenal", "away": "Chelsea", "midfield_shift": 0.1, "home_absent": ["Arsenal FWD1"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["midfield_shift"] == 0.1 and body["midfield_sensitivity"]
    assert analytics_client.get("/simulate", params={"home": "Arsenal", "away": "Nowhere"}).status_code == 422
