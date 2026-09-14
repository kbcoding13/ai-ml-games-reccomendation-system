import pytest
from fastapi.testclient import TestClient

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

