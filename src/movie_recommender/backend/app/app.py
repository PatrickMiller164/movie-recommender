from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from movie_recommender.backend.app.recommendations import router as recommendations_router


MODULE_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = MODULE_ROOT / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / 'index.html'

if not FRONTEND_DIR.exists():
    raise RuntimeError(f"Frontend directory {FRONTEND_DIR} does not exist")

app = FastAPI(title="Movie Recommender API")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
app.include_router(recommendations_router)

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_INDEX)
