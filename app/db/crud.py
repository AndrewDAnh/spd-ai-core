from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from typing import List, Optional
import json

from app.db.models import Prediction, ReferenceBaseline


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

