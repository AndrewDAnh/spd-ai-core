from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, UTC
from typing import Dict

from app.models.schemas import (
    BatchValidationRequest,
    BatchValidationResponse,
    ValidationSummary,
    EngineValidationResult,
    QualityIssue,
    ReferenceDataRequest,
    ModelDriftRequest,
    ModelDriftResponse,
    ValidationMetrics
)
from app.core.database import get_db
from app.db import crud
from app.services.data_drift import DataDriftDetector
from app.services.quality_checker import QualityChecker
from app.services.model_drift import ModelDriftDetector
from app.core.config import get_settings
from app.core.logging_config import logger
from app.utils.metrics import aggregate_drift_scores

router = APIRouter()
settings = get_settings()

# In-memory storage for reference data (per plan specification)
reference_store: Dict[str, list] = {}

# Validation metrics tracking
validation_metrics = {
    'total_validations': 0,
    'recent_drift_detections': 0,
    'recent_quality_issues': 0,
    'engines_monitored': set()
}


@router.post("/validate/batch", response_model=BatchValidationResponse)
async def validate_batch(
    request: BatchValidationRequest,
    db: Session = Depends(get_db)
):
    """
    Batch validation endpoint for multiple engines
    
    Performs both drift detection and quality checks on data from multiple engines.
    """
    try:
        # Initialize services
        config = request.config or {}
        drift_threshold = config.drift_threshold if hasattr(config, 'drift_threshold') else settings.DRIFT_THRESHOLD
        outlier_sensitivity = config.outlier_sensitivity if hasattr(config, 'outlier_sensitivity') else settings.OUTLIER_SENSITIVITY
        
        drift_detector = DataDriftDetector(threshold=drift_threshold)
        quality_checker = QualityChecker(outlier_sensitivity=outlier_sensitivity)
        
        engine_results = []
        all_drift_scores = {}
        total_issues = 0
        high_severity = 0
        medium_severity = 0
        drift_detected_count = 0
        
        for engine_data in request.engines:
            engine_id = engine_data.engine_id
            current_data = engine_data.data
            
            # Track engine
            validation_metrics['engines_monitored'].add(engine_id)
            
            # Get reference data
            if request.use_stored_reference:
                reference_data = reference_store.get(engine_id)
                if not reference_data:
                    # Try to get from database
                    baseline = crud.get_baseline(db, engine_id)
                    if baseline:
                        reference_data = baseline.get('data', [])
                    else:
                        # No reference available, skip drift detection
                        reference_data = None
            else:
                reference_data = None
            
            # Quality checks (always performed)
            quality_result = quality_checker.check_quality(current_data)
            
            # Drift detection (only if reference available)
            drift_detected = False
            drift_score = None
            
            if reference_data and len(reference_data) > 0:
                drift_result = drift_detector.detect_drift(reference_data, current_data)
                drift_detected = drift_result['drift_detected']
                drift_score = drift_result['overall_drift_score']
                
                # Aggregate drift scores
                for feature, drift_info in drift_result['feature_drifts'].items():
                    if feature not in all_drift_scores:
                        all_drift_scores[feature] = []
                    all_drift_scores[feature].append(drift_info['score'])
            
            # Determine status
            if not quality_result['quality_passed'] and drift_detected:
                status = "critical"
            elif not quality_result['quality_passed'] or drift_detected:
                status = "warning"
            else:
                status = "ok"
            
            # Convert issues
            issues = [
                QualityIssue(**issue) for issue in quality_result['issues']
            ]
            
            # Update counters
            total_issues += len(issues)
            high_severity += quality_result['summary']['high_severity']
            medium_severity += quality_result['summary']['medium_severity']
            if drift_detected:
                drift_detected_count += 1
            
            engine_results.append(EngineValidationResult(
                engine_id=engine_id,
                status=status,
                drift_detected=drift_detected,
                drift_score=drift_score,
                quality_passed=quality_result['quality_passed'],
                issues=issues if issues else None
            ))
        
        # Calculate detailed stats
        detailed_stats = {}
        if all_drift_scores:
            drift_by_feature = {}
            for feature, scores in all_drift_scores.items():
                drift_by_feature[feature] = {
                    'mean_drift': sum(scores) / len(scores),
                    'max_drift': max(scores)
                }
            detailed_stats['drift_by_feature'] = drift_by_feature
        
        # Create summary
        summary = ValidationSummary(
            total_engines=len(request.engines),
            engines_with_issues=len([r for r in engine_results if r.status != "ok"]),
            high_severity_count=high_severity,
            medium_severity_count=medium_severity,
            drift_detected_count=drift_detected_count
        )
        
        # Update metrics
        validation_metrics['total_validations'] += 1
        validation_metrics['recent_drift_detections'] += drift_detected_count
        validation_metrics['recent_quality_issues'] += total_issues
        
        logger.info(f"Batch validation completed: {request.validation_id}, {len(request.engines)} engines")
        
        return BatchValidationResponse(
            validation_id=request.validation_id,
            timestamp=datetime.now(UTC).isoformat(),
            summary=summary,
            engines=engine_results,
            detailed_stats=detailed_stats if detailed_stats else None
        )
        
    except Exception as e:
        logger.error(f"Error in batch validation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/reference")
async def store_reference(
    request: ReferenceDataRequest,
    db: Session = Depends(get_db)
):
    """
    Store reference baseline for an engine
    
    Stores both in-memory and in database for persistence.
    """
    try:
        # Store in memory
        reference_store[request.engine_id] = request.reference_data
        
        # Also store in database
        baseline_data = {
            'data': request.reference_data,
            'updated_at': datetime.now(UTC).isoformat()
        }
        crud.create_or_update_baseline(
            db=db,
            engine_id=request.engine_id,
            baseline_data=baseline_data
        )
        
        logger.info(f"Reference baseline stored for engine: {request.engine_id}")
        
        return {
            "status": "success",
            "message": f"Reference baseline stored for engine {request.engine_id}",
            "data_points": len(request.reference_data)
        }
        
    except Exception as e:
        logger.error(f"Error storing reference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/drift")
async def validate_drift(
    request: BatchValidationRequest,
    db: Session = Depends(get_db)
):
    """
    Drift detection endpoint.
    Performs drift detection on the data of multiple engines.
    """
    try:
        config = request.config or {}
        drift_threshold = config.drift_threshold if hasattr(config, 'drift_threshold') else settings.DRIFT_THRESHOLD
        
        drift_detector = DataDriftDetector(threshold=drift_threshold)
        
        results = []
        
        for engine_data in request.engines:
            engine_id = engine_data.engine_id
            current_data = engine_data.data
            
            # Get reference data
            if request.use_stored_reference:
                reference_data = reference_store.get(engine_id)
                if not reference_data:
                    baseline = crud.get_baseline(db, engine_id)
                    if baseline:
                        reference_data = baseline.get('data', [])
                    else:
                        reference_data = None
            else:
                reference_data = None
            
            if not reference_data:
                results.append({
                    'engine_id': engine_id,
                    'error': 'No reference data available'
                })
                continue
            
            # Perform drift detection
            drift_result = drift_detector.detect_drift(reference_data, current_data)
            drift_result['engine_id'] = engine_id
            results.append(drift_result)
        
        return {
            'validation_id': request.validation_id,
            'timestamp': datetime.now(UTC).isoformat(),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/quality")
async def validate_quality(request: BatchValidationRequest):
    """
    Quality checks only.
    Performs quality checks on the data of multiple engines.
    """
    try:
        config = request.config or {}
        outlier_sensitivity = config.outlier_sensitivity if hasattr(config, 'outlier_sensitivity') else settings.OUTLIER_SENSITIVITY
        
        quality_checker = QualityChecker(outlier_sensitivity=outlier_sensitivity)
        
        results = []
        
        for engine_data in request.engines:
            quality_result = quality_checker.check_quality(engine_data.data)
            quality_result['engine_id'] = engine_data.engine_id
            results.append(quality_result)
        
        return {
            'validation_id': request.validation_id,
            'timestamp': datetime.now(UTC).isoformat(),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in quality check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/model-drift", response_model=ModelDriftResponse)
async def validate_model_drift(
    request: ModelDriftRequest,
    db: Session = Depends(get_db)
):
    """
    Model drift detection endpoint
    
    Analyzes prediction stability by tracking how much RUL predictions change
    over time for each engine.
    """
    try:
        model_drift_detector = ModelDriftDetector(threshold=request.threshold)
        
        result = model_drift_detector.detect_model_drift(
            db=db,
            engine_ids=request.engines,
            lookback_hours=request.lookback_hours
        )
        
        logger.info(f"Model drift detection completed for {len(request.engines)} engines")
        
        return ModelDriftResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in model drift detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/summary", response_model=ValidationMetrics)
async def get_validation_summary():
    """
    Get validation summary statistics
    
    Returns overall metrics about validation operations.
    """
    return ValidationMetrics(
        total_validations=validation_metrics['total_validations'],
        recent_drift_detections=validation_metrics['recent_drift_detections'],
        recent_quality_issues=validation_metrics['recent_quality_issues'],
        engines_monitored=len(validation_metrics['engines_monitored'])
    )
