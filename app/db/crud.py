from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from typing import List, Optional
import json

from app.db.models import (
    Prediction,
    ReferenceBaseline,
    ModelPerformance,
    ModelRegistry,
    ModelTrainingJob,
    ModelServingConfig,
)


# Prediction CRUD operations

def create_prediction(
    db: Session,
    prediction_id: str,
    batch_id: str,
    engine_id: str,
    prediction_time: datetime,
    remaining_useful_life: float,
    is_going_to_fail: bool,
    confidence: float
) -> Prediction:
    """Create a new prediction record"""
    db_prediction = Prediction(
        prediction_id=prediction_id,
        batch_id=batch_id,
        engine_id=engine_id,
        prediction_time=prediction_time,
        remaining_useful_life=remaining_useful_life,
        is_going_to_fail=is_going_to_fail,
        confidence=confidence
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


def get_predictions_by_engine(
    db: Session,
    engine_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: Optional[int] = None
) -> List[Prediction]:
    """Get predictions for a specific engine"""
    query = db.query(Prediction).filter(Prediction.engine_id == engine_id)
    
    if start_date:
        query = query.filter(Prediction.prediction_time >= start_date)
    if end_date:
        query = query.filter(Prediction.prediction_time <= end_date)
    
    query = query.order_by(desc(Prediction.prediction_time))
    
    if limit:
        query = query.limit(limit)
    
    return query.all()


def get_recent_predictions_by_engine(
    db: Session,
    engine_id: str,
    lookback_hours: int = 24
) -> List[Prediction]:
    """Get recent predictions for an engine within lookback window"""
    cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
    return db.query(Prediction).filter(
        Prediction.engine_id == engine_id,
        Prediction.prediction_time >= cutoff_time
    ).order_by(Prediction.prediction_time).all()


def get_consecutive_predictions(
    db: Session,
    engine_id: str,
    lookback_hours: int = 24
) -> List[Prediction]:
    """Get consecutive predictions for model drift analysis"""
    cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
    return db.query(Prediction).filter(
        Prediction.engine_id == engine_id,
        Prediction.prediction_time >= cutoff_time
    ).order_by(Prediction.prediction_time).all()


# Reference Baseline CRUD operations

def create_or_update_baseline(
    db: Session,
    engine_id: str,
    baseline_data: dict
) -> ReferenceBaseline:
    """Create or update reference baseline for an engine"""
    db_baseline = db.query(ReferenceBaseline).filter(
        ReferenceBaseline.engine_id == engine_id
    ).first()
    
    baseline_json = json.dumps(baseline_data)
    
    if db_baseline:
        db_baseline.baseline_data = baseline_json
        db_baseline.updated_at = datetime.utcnow()
    else:
        db_baseline = ReferenceBaseline(
            engine_id=engine_id,
            baseline_data=baseline_json
        )
        db.add(db_baseline)
    
    db.commit()
    db.refresh(db_baseline)
    return db_baseline


def get_baseline(db: Session, engine_id: str) -> Optional[dict]:
    """Get reference baseline for an engine"""
    db_baseline = db.query(ReferenceBaseline).filter(
        ReferenceBaseline.engine_id == engine_id
    ).first()
    
    if db_baseline:
        return json.loads(db_baseline.baseline_data)
    return None


def create_model_performance(
    db: Session,
    *,
    mean_squared_error: float,
    mean_absolute_error: float,
    mean_absolute_percentage_error: float,
    precision: list,
    recall: list,
    f1_score: float,
    validation_time: datetime,
) -> ModelPerformance:
    """Persist model performance metrics."""
    # Ensure precision and recall are stored as JSON arrays of numbers
    # Convert any integers to floats for consistency
    precision_floats = [float(x) for x in precision] if precision else []
    recall_floats = [float(x) for x in recall] if recall else []
    
    record = ModelPerformance(
        mean_squared_error=mean_squared_error,
        mean_absolute_error=mean_absolute_error,
        mean_absolute_percentage_error=mean_absolute_percentage_error,
        precision=json.dumps(precision_floats),
        recall=json.dumps(recall_floats),
        f1_score=f1_score,
        validation_time=validation_time,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_latest_model_performance(db: Session) -> Optional[ModelPerformance]:
    """Return the latest model performance metrics record if available."""
    return (
        db.query(ModelPerformance)
        .order_by(ModelPerformance.validation_time.desc())
        .first()
    )


# Model Registry CRUD operations

def upsert_model_registry_entry(
    db: Session,
    *,
    model_name: str,
    model_type: str,
    status: str,
    artifact_path: str,
    metrics: Optional[dict] = None,
) -> ModelRegistry:
    payload = {
        "status": status,
        "artifact_path": artifact_path,
        "metrics": json.dumps(metrics) if metrics else None,
    }

    record = db.query(ModelRegistry).filter(ModelRegistry.model_name == model_name).first()
    if record:
        record.status = payload["status"]
        record.artifact_path = payload["artifact_path"]
        record.metrics = payload["metrics"]
    else:
        record = ModelRegistry(
            model_name=model_name,
            model_type=model_type,
            status=payload["status"],
            artifact_path=payload["artifact_path"],
            metrics=payload["metrics"],
        )
        db.add(record)

    db.commit()
    db.refresh(record)
    return record


def list_model_registry(db: Session) -> List[ModelRegistry]:
    return db.query(ModelRegistry).order_by(ModelRegistry.updated_at.desc()).all()


def get_model_registry_entry(db: Session, model_name: str) -> Optional[ModelRegistry]:
    return db.query(ModelRegistry).filter(ModelRegistry.model_name == model_name).first()


def update_model_registry_status(
    db: Session,
    *,
    model_name: str,
    status: str,
) -> Optional[ModelRegistry]:
    record = db.query(ModelRegistry).filter(ModelRegistry.model_name == model_name).first()
    if record is None:
        return None

    record.status = status
    db.commit()
    db.refresh(record)
    return record


# Model Training Job CRUD operations

def create_training_job(
    db: Session,
    *,
    job_id: str,
    requested_models: List[str],
    dataset_partition: Optional[str],
    dataset_metadata: Optional[dict],
) -> ModelTrainingJob:
    job = ModelTrainingJob(
        job_id=job_id,
        requested_models=",".join(requested_models),
        status="queued",
        progress=0.0,
        progress_message="Queued",
        dataset_partition=dataset_partition,
        dataset_metadata=json.dumps(dataset_metadata) if dataset_metadata else None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def update_training_job(
    db: Session,
    *,
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    progress_message: Optional[str] = None,
    metrics: Optional[dict] = None,
    artifact_paths: Optional[dict] = None,
) -> Optional[ModelTrainingJob]:
    job = db.query(ModelTrainingJob).filter(ModelTrainingJob.job_id == job_id).first()
    if job is None:
        return None

    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = progress
    if progress_message is not None:
        job.progress_message = progress_message
    if metrics is not None:
        job.metrics = json.dumps(metrics)
    if artifact_paths is not None:
        job.artifact_paths = json.dumps(artifact_paths)

    db.commit()
    db.refresh(job)
    return job


def get_training_job(db: Session, job_id: str) -> Optional[ModelTrainingJob]:
    return db.query(ModelTrainingJob).filter(ModelTrainingJob.job_id == job_id).first()


# Serving configuration

def get_serving_config(db: Session) -> ModelServingConfig:
    config = db.query(ModelServingConfig).first()
    if config is None:
        config = ModelServingConfig()
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


def update_serving_config(
    db: Session,
    *,
    regression_model: Optional[str] = None,
    classification_model: Optional[str] = None,
) -> ModelServingConfig:
    config = get_serving_config(db)
    if regression_model is not None:
        config.active_regression_model = regression_model
    if classification_model is not None:
        config.active_classification_model = classification_model
    db.commit()
    db.refresh(config)
    return config
