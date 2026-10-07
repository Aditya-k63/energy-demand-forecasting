import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from huggingface_hub import hf_hub_download

from app.config import settings

WEATHER_VARS = ["temperature", "apparentTemperature", "humidity", "windSpeed"]

HOLIDAYS_FILE = Path(__file__).resolve().parent.parent / "data" / "uk_bank_holidays.json"


@lru_cache(maxsize=1)
def _holiday_dates() -> frozenset:
    with open(HOLIDAYS_FILE) as f:
        return frozenset(json.load(f))


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    repo = settings.model_repo
    cfg_path = hf_hub_download(repo, "config.json")
    with open(cfg_path) as f:
        config = json.load(f)

    models = {
        name: lgb.Booster(model_file=hf_hub_download(repo, f"lightgbm_{name}.txt"))
        for name in ["point", "q10", "q50", "q90"]
    }
    replay = pd.read_parquet(hf_hub_download(repo, "replay_test.parquet"))
    replay["hour"] = pd.to_datetime(replay["hour"], utc=True)

    return {
        "config": config,
        "models": models,
        "replay": replay,
        "features": list(config["features"]),
        "lo": float(config["conformal_offsets_kwh"]["lo"]),
        "hi": float(config["conformal_offsets_kwh"]["hi"]),
    }


def _step_features(series: list[float], ts: pd.Timestamp, weather: dict[str, float]) -> dict:
    s = pd.Series(series, dtype=float)
    is_holiday = int(ts.strftime("%Y-%m-%d") in _holiday_dates())
    dow = ts.dayofweek
    is_weekend = int(dow >= 5)
    if is_holiday:
        dow = 6
        is_weekend = 1
    feats = {
        "hour_of_day": ts.hour,
        "dow": dow,
        "month": ts.month,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "lag_1": s.iloc[-1],
        "lag_2": s.iloc[-2],
        "lag_3": s.iloc[-3],
        "lag_24": s.iloc[-24],
        "lag_48": s.iloc[-48],
        "lag_72": s.iloc[-72],
        "lag_168": s.iloc[-168],
        "roll_24_mean": s.iloc[-24:].mean(),
        "roll_24_std": s.iloc[-24:].std(),
        "roll_168_mean": s.iloc[-168:].mean(),
    }
    feats.update(weather)
    return feats


def _weather_at(step: int, weather_forecast, weather_history) -> dict[str, float]:
    out = {}
    for var in WEATHER_VARS:
        if weather_forecast and var in weather_forecast and len(weather_forecast[var]) > step:
            out[var] = float(weather_forecast[var][step])
        elif weather_history and var in weather_history and len(weather_history[var]) >= 24:
            out[var] = float(weather_history[var][step % 24])
        else:
            out[var] = settings.weather_fallback[var]
    return out


def recursive_forecast(
    history: list[float],
    start: pd.Timestamp,
    weather_forecast=None,
    weather_history=None,
    horizon: int | None = None,
) -> list[dict]:
    art = load_artifacts()
    models, features = art["models"], art["features"]
    lo, hi = art["lo"], art["hi"]
    horizon = horizon or settings.horizon_hours

    series = list(history)
    rows = []
    for step in range(horizon):
        ts = start + pd.Timedelta(hours=step)
        weather = _weather_at(step, weather_forecast, weather_history)
        X = pd.DataFrame([_step_features(series, ts, weather)])[features]

        p50 = float(models["q50"].predict(X)[0])
        point = float(models["point"].predict(X)[0])
        series.append(p50)

        rows.append({
            "ts": ts.isoformat(),
            "point": round(point, 5),
            "p50": round(p50, 5),
            "p10": round(p50 + lo, 5),
            "p90": round(p50 + hi, 5),
        })
    return rows


def replay(start: str, days: int) -> tuple[list[dict], dict[str, str]]:
    art = load_artifacts()
    df = art["replay"]
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = t0 + pd.Timedelta(days=days)
    sub = df[(df["hour"] >= t0) & (df["hour"] < t1)]
    rows = [
        {
            "hour": r.hour.isoformat(),
            "demand": round(float(r.demand), 5),
            "point": round(float(r.point), 5),
            "p50": round(float(r.p50), 5),
            "cp_lo": round(float(r.cp_lo), 5),
            "cp_hi": round(float(r.cp_hi), 5),
            "temperature": round(float(r.temperature), 3),
            "apparentTemperature": round(float(r.apparentTemperature), 3),
            "humidity": round(float(r.humidity), 4),
            "windSpeed": round(float(r.windSpeed), 3),
        }
        for r in sub.itertuples()
    ]
    test_range = {
        "min": df["hour"].min().isoformat(),
        "max": df["hour"].max().isoformat(),
    }
    return rows, test_range
