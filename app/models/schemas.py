from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


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


class EngineData(BaseModel):
    """Single engine's time-series data for prediction"""
    engine_id: str
    timestamp: datetime
    data: List[CmapssDataPoint]  # C-MAPSS format data points


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    batch_id: Optional[str] = ""
    engines: List[EngineData]


class PredictionResult(BaseModel):
    """Single prediction result"""
    engine_id: str
    prediction_time: datetime
    remaining_useful_life: float
    is_going_to_fail: Optional[bool]  # None when classifier unavailable
    confidence: float


class BatchPredictionResponse(BaseModel):
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


class BatchValidationResponse(BaseModel):
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


class PredictionChange(BaseModel):
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


class ModelDriftResponse(BaseModel):
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


# Health Check Schema

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str

