import pytest

#Imported settings to get the default demo datasset paths
from app.core.config import settings
from app.services.recommender import Recommender

#Recommender loads files from the instructor

@pytest.fixture
def recommender():
    return Recommender(settings.games_path, settings.neighbors_path)

#Tests are self contained and not depend on FastAPI's lru_cache singleton
def test_search_finds_known_game(recommender):
    results = recommender.search("hades")
    assert any(game.name == "Hades" for game in results)


def test_search_empty_query_returns_empty_list(recommender):
    assert recommender.search("") == []

def test_get_game_returns_none_for_unknown_id(recommender):
    results = recommender.search("hades")
    assert recommender.get_game(999999) is None