from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    history: list[float] = Field(..., min_length=168, description="Hourly demand (kWh), oldest to newest, at least 168 values")
    start: str = Field(..., description="ISO timestamp of the first forecast hour, e.g. 2014-02-15T00:00:00Z")
    weather_forecast: dict[str, list[float]] | None = None
    weather_history: dict[str, list[float]] | None = None


class ForecastPoint(BaseModel):
    ts: str
    point: float
    p50: float
    p10: float
    p90: float


class ForecastResponse(BaseModel):
    horizon_hours: int
    interval: str
    points: list[ForecastPoint]


class ReplayRow(BaseModel):
    hour: str
    demand: float
    point: float
    p50: float
    cp_lo: float
    cp_hi: float
    temperature: float | None = None
    apparentTemperature: float | None = None
    humidity: float | None = None
    windSpeed: float | None = None


class ReplayResponse(BaseModel):
    rows: list[ReplayRow]
    test_range: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    model: str
