
#Imported settings to get the default demo datasset paths
from app.core.config import settings
from app.services.recommender import Recommender

#Reccomender loads files from the instructor


#Tests are self contained and not depend on FastAPI's lru_cache singleton
def test_search_finds_known_game():
    reccomender = Recommender(settings.games_path, settings.neighbors_path)
    results = reccomender.search("hades")
    assert any(game.name == "Hades" for game in results)


def test_search_empty_query_returns_empty_list():
    reccomender = Recommender(settings.games_path, settings.neighbors_path)
    assert reccomender.search("") == []

def test_get_game_returns_none_for_unknown_id():
    reccomender = Recommender(settings.games_path, settings.neighbors_path)
    results = reccomender.search("hades")
    assert any(game.name == "Hades" for game in results)