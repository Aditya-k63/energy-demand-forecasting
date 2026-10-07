from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_repo: str = "neuronsbyisshu/london-energy-demand-forecast-lgbm"
    horizon_hours: int = 24

    weather_fallback: dict[str, float] = {
        "temperature": 9.0,
        "apparentTemperature": 7.5,
        "humidity": 0.82,
        "windSpeed": 4.0,
    }


settings = Settings()
