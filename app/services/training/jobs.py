"""Background job helpers for scheduling and running retraining."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict
from uuid import uuid4

from fastapi import BackgroundTasks, HTTPException

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.logging_config import logger
from app.db import crud
from app.models.schemas import (
    RetrainingRequest,
    TrainingJobStatus,
)
from app.services.model_inference import ModelInferenceService
from app.services.model_performance import ModelPerformanceService
from app.services.training.routines import train_classification_model, train_regression_model


_SETTINGS = get_settings()
_DATA_ROOT = Path("data")
_TRAINING_ROOT = _DATA_ROOT / "training_jobs"
_MODELS_ROOT = Path("models")
_REGRESSION_ROOT = _MODELS_ROOT / "regression"
_CLASSIFICATION_ROOT = _MODELS_ROOT / "classification"


class TrainingJobManager:
    """Helper for scheduling and executing retraining jobs."""

    def __init__(self) -> None:
        _TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
        _REGRESSION_ROOT.mkdir(parents=True, exist_ok=True)
        _CLASSIFICATION_ROOT.mkdir(parents=True, exist_ok=True)

    def schedule_job(
        self,
        *,
        request: RetrainingRequest,
        db_session,
        background_tasks: BackgroundTasks,
    ) -> TrainingJobStatus:
        requested_models = request.requested_model_types()
        if not requested_models:
            raise HTTPException(status_code=400, detail="No model types selected for retraining")

        job_id = request.job_id or f"job_{uuid4().hex[:12]}"
        dataset_snapshot_path = (_TRAINING_ROOT / f"{job_id}.json").resolve()
        self._persist_dataset_snapshot(dataset_snapshot_path, request)

        job = crud.create_training_job(
            db_session,
            job_id=job_id,
            requested_models=requested_models,
            dataset_partition=request.dataset.partition,
            dataset_metadata=request.dataset.metadata,
        )

        background_tasks.add_task(
            self._run_job,
            job_id,
            request.model_dump(mode="json"),
            str(dataset_snapshot_path),
        )

        return self._to_status(job)

    def _persist_dataset_snapshot(self, path: Path, request: RetrainingRequest) -> None:
        payload = request.model_dump(mode="json")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        logger.info("Stored training dataset snapshot at %s", path)

    def _run_job(self, job_id: str, request_payload: Dict[str, object], dataset_path: str) -> None:
        logger.info("Starting retraining job %s (dataset snapshot: %s)", job_id, dataset_path)
        db = SessionLocal()
        try:
            crud.update_training_job(
                db,
                job_id=job_id,
                status="running",
                progress=0.05,
                progress_message="Preparing training artifacts",
            )

            request = RetrainingRequest(**request_payload)
            requested_models = request.requested_model_types()

            artifact_paths: Dict[str, str] = {}
            metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

            active_config = crud.get_serving_config(db)
            current_registry = {entry.model_name: entry for entry in crud.list_model_registry(db)}

            current_regression_path = self._resolve_current_path(
                model_name=active_config.active_regression_model,
                default_path=_SETTINGS.REGRESSION_MODEL_PATH,
                registry=current_registry,
            )
            current_classification_path = self._resolve_current_path(
                model_name=active_config.active_classification_model,
                default_path=_SETTINGS.CLASSIFICATION_MODEL_PATH,
                registry=current_registry,
            )

            crud.update_training_job(
                db,
                job_id=job_id,
                progress=0.15,
                progress_message="Preparing training directories",
            )

            if "regression" in requested_models:
                artifact_paths["regression"] = self._prepare_run_directory(
                    source=current_regression_path,
                    target_root=_REGRESSION_ROOT,
                    job_id=job_id,
                )

            if "classification" in requested_models:
                artifact_paths["classification"] = self._prepare_run_directory(
                    source=current_classification_path,
                    target_root=_CLASSIFICATION_ROOT,
                    job_id=job_id,
                )

            crud.update_training_job(
                db,
                job_id=job_id,
                progress=0.35,
                progress_message="Starting training runs",
                artifact_paths=artifact_paths,
            )

            if "regression" in requested_models:
                crud.update_training_job(
                    db,
                    job_id=job_id,
                    progress=0.45,
                    progress_message="Training regression model",
                )
                training_metrics = train_regression_model(
                    request.dataset,
                    run_dir=artifact_paths["regression"],
                    device=_SETTINGS.DEVICE,
                )
                metrics.setdefault("regression", {})["training"] = training_metrics

            if "classification" in requested_models:
                crud.update_training_job(
                    db,
                    job_id=job_id,
                    progress=0.55,
                    progress_message="Training classification model",
                )
                training_metrics = train_classification_model(
                    request.dataset,
                    run_dir=artifact_paths["classification"],
                    device=_SETTINGS.DEVICE,
                )
                metrics.setdefault("classification", {})["training"] = training_metrics

            crud.update_training_job(
                db,
                job_id=job_id,
                progress=0.7,
                progress_message="Evaluating retrained models",
            )

            if "regression" in requested_models:
                eval_classification_path = artifact_paths.get("classification", current_classification_path)
                regression_metrics = self._evaluate_models(
                    regression_path=artifact_paths.get("regression"),
                    classification_path=eval_classification_path,
                )
                metrics.setdefault("regression", {})["evaluation"] = {
                    "mse": regression_metrics.mean_squared_error,
                    "mae": regression_metrics.mean_absolute_error,
                    "mape": regression_metrics.mean_absolute_percentage_error,
                }
                self._update_registry_entry(
                    db,
                    model_type="regression",
                    artifact_path=artifact_paths["regression"],
                    metrics=metrics["regression"].get("evaluation"),
                    partition=request.dataset.partition,
                    job_id=job_id,
                )

            if "classification" in requested_models:
                eval_regression_path = artifact_paths.get("regression", current_regression_path)
                classification_metrics = self._evaluate_models(
                    regression_path=eval_regression_path,
                    classification_path=artifact_paths.get("classification"),
                )
                metrics.setdefault("classification", {})["evaluation"] = {
                    "f1": classification_metrics.f1_score
                }
                self._update_registry_entry(
                    db,
                    model_type="classification",
                    artifact_path=artifact_paths["classification"],
                    metrics=metrics["classification"].get("evaluation"),
                    partition=request.dataset.partition,
                    job_id=job_id,
                )

            crud.update_training_job(
                db,
                job_id=job_id,
                status="completed",
                progress=1.0,
                progress_message="Retraining completed",
                metrics=metrics,
                artifact_paths=artifact_paths,
            )

            logger.info("Retraining job %s completed", job_id)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Retraining job %s failed", job_id)
            crud.update_training_job(
                db,
                job_id=job_id,
                status="failed",
                progress_message=f"Failed: {exc}",
            )
        finally:
            db.close()

    def _resolve_current_path(
        self,
        *,
        model_name: str | None,
        default_path: str,
        registry: Dict[str, object],
    ) -> str:
        if model_name and model_name in registry:
            entry = registry[model_name]
            return entry.artifact_path  # type: ignore[attr-defined]
        return default_path

    def _prepare_run_directory(self, *, source: str, target_root: Path, job_id: str) -> str:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Model directory at {source_path} not found")

        target_dir = target_root / job_id
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("config.yaml", "normalisation.json"):
            src_file = source_path / filename
            if src_file.exists():
                shutil.copy(src_file, target_dir / filename)

        return str(target_dir)

    def _evaluate_models(self, *, regression_path: str, classification_path: str):
        inference_service = ModelInferenceService(
            regression_run_dir=regression_path,
            classification_run_dir=classification_path,
        )
        performance_service = ModelPerformanceService(model_service=inference_service)
        return performance_service.run_evaluation()

    def _update_registry_entry(
        self,
        db_session,
        *,
        model_type: str,
        artifact_path: str,
        metrics: Dict[str, float] | None,
        partition: str,
        job_id: str,
    ) -> None:
        model_name = f"{model_type}_{partition}_{job_id}"
        crud.upsert_model_registry_entry(
            db_session,
            model_name=model_name,
            model_type=model_type,
            status="ready",
            artifact_path=artifact_path,
            metrics=metrics,
        )

    def _to_status(self, job) -> TrainingJobStatus:
        requested_models = job.requested_models.split(",") if job.requested_models else []
        return TrainingJobStatus(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            progress_message=job.progress_message,
            requested_models=requested_models,
            dataset_partition=job.dataset_partition,
            metrics=json.loads(job.metrics) if job.metrics else None,
            artifact_paths=json.loads(job.artifact_paths) if job.artifact_paths else None,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


def fetch_training_job_status(job, manager: TrainingJobManager) -> TrainingJobStatus:
    return manager._to_status(job)  # pylint: disable=protected-access


manager = TrainingJobManager()
