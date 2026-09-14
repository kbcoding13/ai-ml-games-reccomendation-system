import pytest
from fastapi.testclient import TestClient
from app.core.config import settings
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_search_returns_matching_game(client):
    response = client.get("/api/search", params={"q": "hades"})
    assert response.status_code == 200
    names = [game["name"] for game in response.json()["games"]]
    assert "Hades" in names

def test_get_unknown_game_returns_404(client):
    response = client.get("/api/games/999999")
    assert response.status_code == 404

def test_recommendation_returns_scored_list(client):
    response = client.get("api/games/3/recommendations", params={"top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 3
    assert all("score" in rec and "reasons" in rec for rec in body)

def test_reccomendations_top_k_too_high_returns_422(client):
    response = client.get("/api/games/3/recommendations", params={"top_k": 100})
    assert response.status_code == 422

def test_recommendations_top_k_too_low_returns_422(client):
    response = client.get("/api/games/3/recommendations", params={"top_k": 0})
    assert response.status_code == 422

def test_recommendations_default_top_k_is_used_when_omitted(client):
    response = client.get("/api/games/3/recommendations")
    assert response.status_code == 200
    assert len(response.json()) == settings.default_top_k