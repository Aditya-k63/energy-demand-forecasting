from fastapi import APIRouter

from app.services.forecaster import load_artifacts

router = APIRouter()


@router.get("/stats")
async def stats() -> dict:
    art = load_artifacts()
    cfg = art["config"]
    return {
        "model_version": cfg.get("model_version"),
        "target": cfg.get("target"),
        "horizon_hours": int(cfg.get("horizon_hours", 24)),
        "metrics_test": cfg.get("metrics_test", []),
        "holiday_fix_experiment": cfg.get("holiday_fix_experiment"),
        "conformal": {
            "offsets_kwh": cfg.get("conformal_offsets_kwh"),
            "coverage": cfg.get("conformal_coverage"),
        },
        "split_protocol": cfg.get("split_protocol"),
    }
