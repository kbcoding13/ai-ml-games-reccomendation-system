
#Imported settings to get the default demo datasset paths
from app.core.config import settings
from app.services.recommender import Recommender

#Reccomender loads files from the instructor


#Tests are self contained and not depend on FastAPI's lru_cache singleton
def test_search_finds_known_game():
    reccomender = Recommender(settings.games_path, settings.neighbors_path)
    results = reccomender.search("hades")
    assert any(game.name == "Hades" for game in results)

