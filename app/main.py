from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import BASE_DIR

app = FastAPI(title="Game Recommendation System")
app.include_router(router)
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "frontend" / "index.html")
