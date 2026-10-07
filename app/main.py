from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import forecast, meta
from app.services.forecaster import load_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="London Energy Demand Forecasting",
    description="24h-ahead household demand forecasting with LightGBM and conformal-calibrated prediction intervals.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(forecast.router, prefix="/api/v1", tags=["forecast"])
app.include_router(meta.router, prefix="/api/v1", tags=["meta"])

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
