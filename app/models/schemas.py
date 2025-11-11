from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# Global Pydantic configuration for RFC3339 datetime serialization
class RFC3339BaseModel(BaseModel):
    """Base model with RFC3339 datetime serialization."""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v.tzinfo else v.replace(tzinfo=None).isoformat() + 'Z'
        }
    )


# ============= Inference Schemas =============

class CmapssDataPoint(BaseModel):
    """Single time-step data point in C-MAPSS format"""
    cycle: int
    setting_1: float
    setting_2: float
    setting_3: float
    s1: Optional[float] = None
    s2: Optional[float] = None
    s3: Optional[float] = None
    s4: Optional[float] = None
    s5: Optional[float] = None
    s6: Optional[float] = None
    s7: Optional[float] = None
    s8: Optional[float] = None
    s9: Optional[float] = None
    s10: Optional[float] = None
    s11: Optional[float] = None
    s12: Optional[float] = None
    s13: Optional[float] = None
    s14: Optional[float] = None
    s15: Optional[float] = None
    s16: Optional[float] = None
    s17: Optional[float] = None
    s18: Optional[float] = None
    s19: Optional[float] = None
    s20: Optional[float] = None
    s21: Optional[float] = None


class EngineData(RFC3339BaseModel):
    """Single engine's time-series data for prediction"""
    engine_id: str
    timestamp: datetime
    data: List[CmapssDataPoint]  # C-MAPSS format data points


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    batch_id: Optional[str] = ""
    engines: List[EngineData]


class PredictionResult(RFC3339BaseModel):
    """Single prediction result"""
    engine_id: str
    prediction_time: datetime
    remaining_useful_life: float
    is_going_to_fail: Optional[bool]  # None when classifier unavailable
    confidence: float


class BatchPredictionResponse(RFC3339BaseModel):
    """Batch prediction response"""
    prediction_id: str
    batch_id: str
    timestamp: datetime
    predictions: List[PredictionResult]


class PredictionHistory(BaseModel):
    """Prediction history for an engine"""
    engine_id: str
    predictions: List[PredictionResult]
    total_count: int


# ============= Validation Schemas =============

class ValidationConfig(BaseModel):
    """Configuration for validation checks"""
    drift_threshold: Optional[float] = 0.2
    outlier_sensitivity: Optional[str] = "medium"


class EngineValidationData(BaseModel):
    """Data for validation of a single engine"""
    engine_id: str
    data: List[Dict[str, Optional[float]]]  # Allow None for missing values


class BatchValidationRequest(BaseModel):
    """Batch validation request"""
    validation_id: str
    engines: List[EngineValidationData]
    use_stored_reference: bool = True
    config: Optional[ValidationConfig] = None


class ReferenceDataRequest(BaseModel):
    """Request to store reference baseline"""
    engine_id: str
    reference_data: List[Dict[str, Optional[float]]]  # Allow None for missing values


# Data Drift Schemas

class FeatureDrift(BaseModel):
    """Drift information for a single feature"""
    score: float
    method: str
    status: str
    p_value: Optional[float] = None


class DriftResult(BaseModel):
    """Drift detection result for a single engine"""
    drift_detected: bool
    overall_drift_score: float
    feature_drifts: Dict[str, FeatureDrift]
    threshold: float


# Quality Check Schemas

class QualityIssue(BaseModel):
    """Single quality issue"""
    feature: str
    type: str
    severity: str
    details: str


class QualityResult(BaseModel):
    """Quality check result"""
    quality_passed: bool
    issues: List[QualityIssue]
    summary: Dict[str, int]


# Combined Validation Response

class EngineValidationResult(BaseModel):
    """Validation result for a single engine"""
    engine_id: str
    status: str
    drift_detected: bool
    drift_score: Optional[float] = None
    quality_passed: bool
    issues: Optional[List[QualityIssue]] = None


class ValidationSummary(BaseModel):
    """Summary statistics for batch validation"""
    total_engines: int
    engines_with_issues: int
    high_severity_count: int
    medium_severity_count: int
    drift_detected_count: int


class BatchValidationResponse(RFC3339BaseModel):
    """Batch validation response"""
    validation_id: str
    timestamp: datetime
    summary: ValidationSummary
    engines: List[EngineValidationResult]
    detailed_stats: Optional[Dict[str, Any]] = None


# Model Drift Schemas

class ModelDriftRequest(BaseModel):
    """Request for model drift detection"""
    engines: List[str]
    lookback_hours: int = 24
    threshold: float = 5.0


class PredictionChange(RFC3339BaseModel):
    """Single prediction in time-series"""
    time: datetime
    rul: float
    change_rate: Optional[float] = None


class EngineModelDrift(BaseModel):
    """Model drift result for a single engine"""
    engine_id: str
    status: str
    prediction_count: int
    avg_rul_change_rate: Optional[float] = None
    max_rul_change_rate: Optional[float] = None
    consecutive_predictions: Optional[List[PredictionChange]] = None
    alert: Optional[str] = None


class ModelDriftSummary(BaseModel):
    """Summary of model drift analysis"""
    total_engines: int
    unstable_engines: int
    avg_stability_score: Optional[float] = None


class ModelDriftResponse(RFC3339BaseModel):
    """Response for model drift detection"""
    timestamp: datetime
    summary: ModelDriftSummary
    engines: List[EngineModelDrift]


# Validation Summary Endpoint

class ValidationMetrics(BaseModel):
    """Overall validation metrics"""
    total_validations: int
    recent_drift_detections: int
    recent_quality_issues: int
    engines_monitored: int


class ModelPerformanceMetrics(RFC3339BaseModel):
    """Stored model performance metrics."""

    mean_squared_error: float
    mean_absolute_error: float
    mean_absolute_percentage_error: float
    precision: List[float] = Field(default_factory=list)
    recall: List[float] = Field(default_factory=list)
    f1_score: float = Field(default_factory=float)
    validation_time: datetime


# ============= Continuous Training Schemas =============


class RegressionRetrainSample(BaseModel):
    """Single regression training sample containing full sequence and target RUL."""

    engine_id: str
    data: List[CmapssDataPoint]
    target_rul: float


class ClassificationRetrainSample(BaseModel):
    """Single classification training sample containing sequence and binary label."""

    engine_id: str
    data: List[CmapssDataPoint]
    label: int


class RetrainingDataset(BaseModel):
    """Dataset payload supplied by the web backend for retraining."""

    partition: str
    regression_samples: Optional[List[RegressionRetrainSample]] = None
    classification_samples: Optional[List[ClassificationRetrainSample]] = None
    metadata: Optional[Dict[str, Any]] = None


class RetrainingRequest(BaseModel):
    """Request body for triggering model retraining."""

    job_id: Optional[str] = None
    retrain_regression: bool = False
    retrain_classification: bool = False
    dataset: RetrainingDataset

    def requested_model_types(self) -> List[str]:
        requested: List[str] = []
        if self.retrain_regression:
            requested.append("regression")
        if self.retrain_classification:
            requested.append("classification")
        return requested


class TrainingJobStatus(RFC3339BaseModel):
    """Response describing the state of a retraining job."""

    job_id: str
    status: str
    progress: Optional[float] = None
    progress_message: Optional[str] = None
    requested_models: List[str]
    dataset_partition: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    artifact_paths: Optional[Dict[str, str]] = None
    created_at: datetime
    updated_at: datetime


class TrainingJobUpdateRequest(BaseModel):
    """Payload from the web backend to push manual job progress updates."""

    status: Optional[str] = None
    progress: Optional[float] = None
    progress_message: Optional[str] = None


class ModelRegistryEntry(RFC3339BaseModel):
    """Single entry in the model registry."""

    model_name: str
    model_type: str
    status: str
    artifact_path: str
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class ModelRegistryResponse(BaseModel):
    """Response listing all models available in the registry."""

    models: List[ModelRegistryEntry]


class ModelSelectionRequest(BaseModel):
    """Request to select regression and classification models for serving."""

    regression_model: Optional[str] = None
    classification_model: Optional[str] = None


class ModelSelectionResponse(RFC3339BaseModel):
    """Response after updating active serving models."""

    active_regression_model: Optional[str]
    active_classification_model: Optional[str]
    updated_at: datetime


# Health Check Schema

class HealthResponse(RFC3339BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str

