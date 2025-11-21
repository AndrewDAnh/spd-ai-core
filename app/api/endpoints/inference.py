from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, UTC
from typing import Optional
import uuid

from app.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResult,
    PredictionHistory
)
from app.core.database import get_db
from app.db import crud
from app.services.model_inference import ModelInferenceService
from app.core.logging_config import logger

router = APIRouter()

# Initialize model service (singleton)
model_service = ModelInferenceService()


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Batch prediction endpoint for multiple engines
    
    Accepts time-series data for multiple engines and returns RUL predictions.
    Stores all predictions in the database for drift tracking.
    """
    try:
        # Generate unique prediction ID
        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(UTC)
        
        predictions = []
        
        for engine_data in request.engines:
            # Make prediction
            rul, is_going_to_fail, confidence = model_service.predict(
                engine_id=engine_data.engine_id,
                data=engine_data.data,
                timestamp=engine_data.timestamp
            )
            
            # Store in database
            crud.create_prediction(
                db=db,
                prediction_id=prediction_id,
                batch_id=request.batch_id,
                engine_id=engine_data.engine_id,
                prediction_time=timestamp,
                remaining_useful_life=rul,
                is_going_to_fail=is_going_to_fail,
                confidence=confidence
            )
            
            # Add to response
            predictions.append(PredictionResult(
                engine_id=engine_data.engine_id,
                prediction_time=timestamp,
                remaining_useful_life=rul,
                is_going_to_fail=is_going_to_fail,
                confidence=confidence
            ))
        
        logger.info(f"Batch prediction completed: {prediction_id}, {len(predictions)} engines")
        
        return BatchPredictionResponse(
            prediction_id=prediction_id,
            batch_id=request.batch_id,
            timestamp=timestamp,
            predictions=predictions
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/history/{engine_id}", response_model=PredictionHistory)
async def get_prediction_history(
    engine_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: Optional[int] = Query(100),
    db: Session = Depends(get_db)
):
    """
    Get prediction history for a specific engine
    
    Query parameters:
    - start_date: Filter predictions after this date
    - end_date: Filter predictions before this date
    - limit: Maximum number of predictions to return (default: 100)
    """
    try:
        predictions = crud.get_predictions_by_engine(
            db=db,
            engine_id=engine_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        prediction_results = [
            PredictionResult(
                engine_id=pred.engine_id,
                prediction_time=pred.prediction_time,
                remaining_useful_life=pred.remaining_useful_life,
                is_going_to_fail=pred.is_going_to_fail,
                confidence=pred.confidence
            )
            for pred in predictions
        ]
        
        return PredictionHistory(
            engine_id=engine_id,
            predictions=prediction_results,
            total_count=len(prediction_results)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
