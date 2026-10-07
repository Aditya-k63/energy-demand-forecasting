from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

import pandas as pd

from app.config import settings
from app.schemas import ForecastRequest, ForecastResponse, HealthResponse, ReplayResponse
from app.services.forecaster import recursive_forecast, replay

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=settings.model_repo)


@router.get("/replay", response_model=ReplayResponse)
async def replay_endpoint(start: str = "2014-01-13", days: int = 7):
    if not (1 <= days <= 59):
        raise HTTPException(status_code=422, detail="days must be between 1 and 59")
    try:
        rows, test_range = replay(start, days)
    except (ValueError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=f"invalid start date: {e}")
    if not rows:
        raise HTTPException(status_code=404, detail=f"no replay data in range; test range is {test_range['min']} → {test_range['max']}")
    return ReplayResponse(rows=rows, test_range=test_range)


@router.post("/forecast", response_model=ForecastResponse)
async def forecast_endpoint(request: ForecastRequest) -> ForecastResponse:
    try:
        start = pd.Timestamp(request.start)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
    except Exception:
        raise HTTPException(status_code=422, detail="invalid start timestamp")
    points = recursive_forecast(
        history=request.history,
        start=start,
        weather_forecast=request.weather_forecast,
        weather_history=request.weather_history,
    )
    return ForecastResponse(
        horizon_hours=len(points),
        interval="80% conformal-calibrated",
        points=points,
    )
