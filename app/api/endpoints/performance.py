from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json

from app.core.database import get_db
from app.core.logging_config import logger
from app.db import crud
from app.db.models import ModelPerformance
from app.models.schemas import ModelPerformanceMetrics
from app.services.model_performance import model_performance_service

router = APIRouter()


def _record_to_schema(record: ModelPerformance) -> ModelPerformanceMetrics:
    return ModelPerformanceMetrics(
        mean_squared_error=record.mean_squared_error,
        mean_absolute_error=record.mean_absolute_error,
        mean_absolute_percentage_error=record.mean_absolute_percentage_error,
        precision=json.loads(record.precision) if record.precision else [],
        recall=json.loads(record.recall) if record.recall else [],
        f1_score=json.loads(record.f1_score) if record.f1_score else 0.0,
        validation_time=record.validation_time,
    )


@router.post("/models/performance/run", response_model=ModelPerformanceMetrics)
async def run_model_performance(db: Session = Depends(get_db)) -> ModelPerformanceMetrics:
    """Run the STAR regression model on the FD001 test split and persist metrics."""
    try:
        result = model_performance_service.run_evaluation()
        record = crud.create_model_performance(db, **result.to_dict())
        logger.info("Model performance evaluation completed and stored.")
        return _record_to_schema(record)
    except Exception as exc:  # pragma: no cover - FastAPI handles response
        logger.error(f"Model performance evaluation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models/performance", response_model=ModelPerformanceMetrics)
async def get_model_performance(db: Session = Depends(get_db)) -> ModelPerformanceMetrics:
    """Return the most recently computed performance metrics."""
    record = crud.get_latest_model_performance(db)
    if record is None:
        raise HTTPException(status_code=404, detail="Model performance metrics not found")
    return _record_to_schema(record)
