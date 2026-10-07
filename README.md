# London Energy Demand Forecasting

![App screenshot](docs/screenshot.png)

24-hour-ahead forecasting of **aggregate hourly household electricity demand** — LightGBM trained on the Smart Meters in London panel, with **conformal-calibrated prediction intervals**, served via FastAPI + Docker.

- Models: [neuronsbyisshu/london-energy-demand-forecast-lgbm](https://huggingface.co/neuronsbyisshu/london-energy-demand-forecast-lgbm)
- Dataset: [Smart Meters in London](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) (Kaggle)

### Uncertainty calibration (headline experiment)

| Method | Nominal coverage | Empirical coverage |
|---|---:|---:|
| Quantile LightGBM (P10–P90) | 80% | 64.3% |
| **+ split-conformal calibration** | **80%** | **82.6%** |

Quantile regression alone significantly under-covered the held-out test set; split-conformal calibration on **validation** residuals corrected the interval to approximately its nominal level. Conformal offsets never touch test data — calibration is validation-only, coverage is reported on test.

## Data sources

| Source | Role |
|---|---|
| Smart Meters in London (Kaggle) | 5,566 households, half-hourly electricity readings, Nov 2011 – Feb 2014 |
| Dark Sky weather (bundled with the dataset) | hourly London temperature, apparent temperature, humidity, wind speed |
| UK bank holidays (bundled with the dataset) | holiday calendar features |

**What is actually forecast:** not 4,443 individual households, but a **single aggregate series**:

```
Raw readings (167M rows, 112 half-hourly block files)
→ Std-tariff household subset (4,443; ToU households excluded — dynamic pricing confounds behavior)
→ per-household hourly energy
→ mean across active households per hour
→ one aggregate hourly demand series (19,864 hours, zero gaps)
→ 24-hour-ahead forecasting
```

## Results — held-out test (Jan–Feb 2014, time-based split)

| Model | MAE | RMSE | sMAPE | vs naive |
|---|---:|---:|---:|---:|
| Seasonal-naive (t-168) | 0.0243 | 0.0372 | 5.05% | — |
| Holt-Winters (rejected) | 0.0518 | 0.0661 | 11.76% | +113.3% |
| **LightGBM v2** | **0.0109** | **0.0153** | **2.33%** | **−55.1%** |

All values measured on the same test window. Why LightGBM wins: the lag/rolling/calendar/weather feature space lets boosted trees capture the nonlinear daily/weekly/temperature structure directly — no sequence model needed, and Holt-Winters (daily seasonality only) can't represent the weekly pattern at all.

## Forecasting protocol

The model is evaluated in a **direct 24h-ahead** setup: each test prediction uses true historical lag values available at prediction time. The deployed `/forecast` endpoint performs **recursive** inference, where each predicted hour feeds back as a lag input for the next — so live recursive error compounds over the 24 steps and reads higher than the offline test MAE. The two are not treated as equivalent metrics, and both are reported.

## Error analysis → controlled fix (v1 → v2)

Phase 4 span-level analysis found the dominant failure mode was **holidays** (MAE 2.4× baseline; only ~20 holiday examples exist, so a binary flag can't teach the behavior). Peak hours, by contrast, showed no degradation.

Controlled experiment (identical params/seed, validation as primary judge): **holiday-as-Sunday encoding** (`dow=6, is_weekend=1` for holiday rows).

| Split | Metric | v1 | v2 | Δ |
|---|---|---:|---:|---:|
| validation | holiday MAE | 0.0351 | 0.0312 | −11.1% |
| test | holiday MAE | 0.0262 | 0.0208 | **−20.6%** |
| test | overall MAE | 0.0111 | 0.0109 | −2.1% |

Adopted as v2. The serving feature pipeline applies the identical encoding (`app/data/uk_bank_holidays.json`), verified against the v2 training pipeline. Experiment record: `notebooks/05_holiday_fix_experiment.ipynb`.

## Pipeline

```
Smart meters + bundled weather
        ↓
Aggregate hourly demand series
        ↓
EDA & seasonality analysis (ACF-validated lag structure)
        ↓
Lag / rolling / calendar / weather features
        ↓
Seasonal-Naive → Holt-Winters → LightGBM
        ↓
Time-based holdout evaluation (MAE / RMSE / sMAPE)
        ↓
Error analysis → holiday fix (v2)
        ↓
Quantile LightGBM (P10/P50/P90)
        ↓
Split-conformal calibration
        ↓
FastAPI + Docker (models pulled from Hugging Face Hub)
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | liveness |
| `GET /api/v1/stats` | metrics, conformal config, split protocol |
| `GET /api/v1/replay?start=YYYY-MM-DD&days=N` | held-out test predictions + actuals (demo replay) |
| `POST /api/v1/forecast` | recursive 24h forecast with conformal interval |

`POST /api/v1/forecast` body:

```json
{
  "history": [0.31, 0.28, "…168+ hourly demand values…"],
  "start": "2014-02-15T00:00:00Z",
  "weather_history": {"temperature": ["…24 values…"]}
}
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 — replay + live forecast UI. API docs at `/docs`.

## Docker

```bash
docker build -t energy-demand-forecast .
docker run -p 8000:8000 energy-demand-forecast
```

Models are pulled from Hugging Face Hub at container startup — the image stays small. Every push to `main` rebuilds and pushes `DOCKER_USERNAME/energy-demand-forecast:latest` via GitHub Actions.

## Notebooks (`notebooks/`)

| Notebook | Contents |
|---|---|
| `01_data_validation.ipynb` | dataset inspection: schema, 167M-row coverage scan, panel completeness |
| `02_eda_and_demand_patterns.ipynb` | aggregate series construction + seasonality/weather EDA |
| `03_feature_engineering_and_model_comparison.ipynb` | features, time split, Naive/Holt-Winters/LightGBM |
| `04_error_analysis_and_conformal.ipynb` | failure-mode analysis, quantile intervals, conformal calibration |
| `05_holiday_fix_experiment.ipynb` | controlled A/B: original vs holiday-as-Sunday encoding |
| `06_hf_model_upload.ipynb` | Hugging Face publication (v2 model set, config, model card) |

## Known limitations

- Holiday MAE still ~2× normal after the fix (few training examples — documented, not hidden)
- Weather inputs assume a perfect weather forecast at prediction time
- Trained on 2013 London data; regime shifts need retraining
- Holiday calendar bundled with the app covers 2012–2014; serving outside that range treats all days as non-holidays
