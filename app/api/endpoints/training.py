from __future__ import annotations

import json
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.logging_config import logger
from app.db import crud
from app.models.schemas import (
    ModelRegistryEntry,
    ModelRegistryResponse,
    ModelSelectionRequest,
    ModelSelectionResponse,
    RetrainingRequest,
    TrainingJobStatus,
    TrainingJobUpdateRequest,
)
from app.services.training import fetch_training_job_status, manager

from app.core.config import get_settings

router = APIRouter()
_settings = get_settings()


def _to_registry_entry(record) -> ModelRegistryEntry:
    metrics = None
    if record.metrics:
        try:
            metrics = json.loads(record.metrics)
        except ValueError:
            metrics = None
    return ModelRegistryEntry(
        model_name=record.model_name,
        model_type=record.model_type,
        status=record.status,
        artifact_path=record.artifact_path,
        metrics=metrics,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _ensure_baseline_entries(db: Session) -> Dict[str, object]:
    entries = crud.list_model_registry(db)
    present = {entry.model_name for entry in entries}
    if "regression_default" not in present:
        crud.upsert_model_registry_entry(
            db,
            model_name="regression_default",
            model_type="regression",
            status="ready",
            artifact_path=_settings.REGRESSION_MODEL_PATH,
            metrics=None,
        )
    if "classification_default" not in present:
        crud.upsert_model_registry_entry(
            db,
            model_name="classification_default",
            model_type="classification",
            status="ready",
            artifact_path=_settings.CLASSIFICATION_MODEL_PATH,
            metrics=None,
        )
    return {entry.model_name: entry for entry in crud.list_model_registry(db)}


def _resolve_artifact_path(db: Session, model_name: str, model_type: str) -> str:
    record = crud.get_model_registry_entry(db, model_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in registry")
    if record.model_type != model_type:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not of type '{model_type}'")
    if record.status not in {"ready", "selected"}:
        raise HTTPException(status_code=409, detail=f"Model '{model_name}' is not ready for serving")
    return record.artifact_path


@router.post("/models/retrain", response_model=TrainingJobStatus)
async def trigger_retraining(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainingJobStatus:
    registry_map = _ensure_baseline_entries(db)
    logger.info("Registry contains %d models prior to retraining", len(registry_map))
    status = manager.schedule_job(request=request, db_session=db, background_tasks=background_tasks)
    return status


@router.get("/models/retrain/{job_id}", response_model=TrainingJobStatus)
async def get_retraining_status(job_id: str, db: Session = Depends(get_db)) -> TrainingJobStatus:
    job = crud.get_training_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found")
    return fetch_training_job_status(job, manager)


@router.patch("/models/retrain/{job_id}", response_model=TrainingJobStatus)
async def update_retraining_status(
    job_id: str,
    update: TrainingJobUpdateRequest,
    db: Session = Depends(get_db),
) -> TrainingJobStatus:
    job = crud.update_training_job(
        db,
        job_id=job_id,
        status=update.status,
        progress=update.progress,
        progress_message=update.progress_message,
    )
    if job is None:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found")
    return fetch_training_job_status(job, manager)


@router.get("/models/registry", response_model=ModelRegistryResponse)
async def list_models(db: Session = Depends(get_db)) -> ModelRegistryResponse:
    entries_map = _ensure_baseline_entries(db)
    entries = [_to_registry_entry(entries_map[name]) for name in sorted(entries_map.keys())]
    return ModelRegistryResponse(models=entries)


@router.post("/models/selection", response_model=ModelSelectionResponse)
async def update_model_selection(
    request: ModelSelectionRequest,
    db: Session = Depends(get_db),
) -> ModelSelectionResponse:
    config = crud.get_serving_config(db)
    previous_regression = config.active_regression_model
    previous_classification = config.active_classification_model

    regression_name = request.regression_model or config.active_regression_model or "regression_default"
    classification_name = (
        request.classification_model or config.active_classification_model or "classification_default"
    )

    regression_path = _resolve_artifact_path(db, regression_name, "regression")
    classification_path = _resolve_artifact_path(db, classification_name, "classification")

    # Reload inference singleton with new artifact paths
    from app.api.endpoints.inference import model_service  # Local import to avoid circular dependency
    from app.services.model_performance import model_performance_service

    logger.info(
        "Reloading inference service with regression=%s classification=%s",
        regression_name,
        classification_name,
    )
    try:
        model_service.reload_models(
            regression_run_dir=regression_path,
            classification_run_dir=classification_path,
        )
        model_performance_service.attach_model_service(model_service)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to reload inference models")
        raise HTTPException(status_code=500, detail="Failed to reload inference models") from exc

    config = crud.update_serving_config(
        db,
        regression_model=regression_name,
        classification_model=classification_name,
    )

    if previous_regression and previous_regression != regression_name:
        crud.update_model_registry_status(db, model_name=previous_regression, status="ready")
    if previous_classification and previous_classification != classification_name:
        crud.update_model_registry_status(db, model_name=previous_classification, status="ready")

    crud.update_model_registry_status(db, model_name=regression_name, status="selected")
    crud.update_model_registry_status(db, model_name=classification_name, status="selected")

    return ModelSelectionResponse(
        active_regression_model=config.active_regression_model,
        active_classification_model=config.active_classification_model,
        updated_at=config.updated_at,
    )
