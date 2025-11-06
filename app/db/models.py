from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base


class Prediction(Base):
    """Prediction table for storing engine RUL predictions"""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String, nullable=False)
    batch_id = Column(String, nullable=False)
    engine_id = Column(String, nullable=False, index=True)
    prediction_time = Column(DateTime, nullable=False, index=True)
    remaining_useful_life = Column(Float, nullable=False)
    is_going_to_fail = Column(Boolean, nullable=True)  # Nullable when classifier unavailable
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class ReferenceBaseline(Base):
    """Reference baseline table for storing reference data statistics"""
    
    __tablename__ = "reference_baselines"
    
    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(String, unique=True, nullable=False, index=True)
    baseline_data = Column(Text, nullable=False)  # JSON serialized
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ModelPerformance(Base):
    """Stores evaluation metrics computed on the held-out test set."""

    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, index=True)
    mean_squared_error = Column(Float, nullable=True)
    mean_absolute_error = Column(Float, nullable=True)
    mean_absolute_percentage_error = Column(Float, nullable=True)
    precision = Column(Text, nullable=True)  # JSON serialized list
    recall = Column(Text, nullable=True)     # JSON serialized list
    f1_score = Column(Float, nullable=True)   # JSON serialized list
    validation_time = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())


class ModelRegistry(Base):
    """Registry storing metadata about available model artifacts."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True, nullable=False)
    model_type = Column(String, nullable=False)  # regression | classification
    status = Column(String, nullable=False, default="ready")
    artifact_path = Column(String, nullable=False)
    metrics = Column(Text, nullable=True)  # JSON serialized metrics summary
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ModelTrainingJob(Base):
    """Tracks asynchronous retraining jobs and associated datasets."""

    __tablename__ = "model_training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, nullable=False, index=True)
    requested_models = Column(String, nullable=False)  # comma-separated model types
    status = Column(String, nullable=False, default="queued")
    progress = Column(Float, nullable=True)
    progress_message = Column(String, nullable=True)
    dataset_partition = Column(String, nullable=True)
    dataset_metadata = Column(Text, nullable=True)  # JSON snapshot of payload metadata
    artifact_paths = Column(Text, nullable=True)  # JSON {model_type: path}
    metrics = Column(Text, nullable=True)  # JSON {model_type: metrics}
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ModelServingConfig(Base):
    """Stores currently serving regression and classification models."""

    __tablename__ = "model_serving_config"

    id = Column(Integer, primary_key=True, index=True)
    active_regression_model = Column(String, nullable=True)
    active_classification_model = Column(String, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
