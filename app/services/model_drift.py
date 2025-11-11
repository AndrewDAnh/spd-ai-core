from sqlalchemy.orm import Session
from datetime import datetime, UTC
from typing import List, Dict
import numpy as np

from app.db import crud
from app.core.logging_config import logger


class ModelDriftDetector:
    """Service for detecting model drift through prediction stability analysis"""
    
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
    
    def detect_model_drift(
        self,
        db: Session,
        engine_ids: List[str],
        lookback_hours: int = 24
    ) -> Dict:
        """
        Detect model drift by analyzing prediction stability
        
        Args:
            db: Database session
            engine_ids: List of engine IDs to analyze
            lookback_hours: Hours to look back for predictions
            
        Returns:
            Dictionary with model drift results
        """
        try:
            engine_results = []
            stability_scores = []
            unstable_count = 0
            
            for engine_id in engine_ids:
                # Get consecutive predictions
                predictions = crud.get_consecutive_predictions(
                    db, engine_id, lookback_hours
                )
                
                if len(predictions) < 2:
                    # Not enough predictions for analysis
                    engine_results.append({
                        'engine_id': engine_id,
                        'status': 'insufficient_data',
                        'prediction_count': len(predictions),
                        'avg_rul_change_rate': None,
                        'max_rul_change_rate': None,
                        'alert': 'Less than 2 predictions available'
                    })
                    continue
                
                # Calculate prediction stability
                drift_result = self._analyze_prediction_stability(
                    predictions,
                    engine_id
                )
                
                engine_results.append(drift_result)
                
                if drift_result['avg_rul_change_rate'] is not None:
                    stability_scores.append(drift_result['avg_rul_change_rate'])
                    
                    if drift_result['status'] == 'unstable':
                        unstable_count += 1
            
            # Calculate summary
            avg_stability = np.mean(stability_scores) if stability_scores else None
            
            summary = {
                'total_engines': len(engine_ids),
                'unstable_engines': unstable_count,
                'avg_stability_score': float(avg_stability) if avg_stability else None
            }
            
            return {
                'timestamp': datetime.now(UTC).isoformat(),
                'summary': summary,
                'engines': engine_results
            }
            
        except Exception as e:
            logger.error(f"Error in model drift detection: {str(e)}")
            raise
    
    def _analyze_prediction_stability(
        self,
        predictions: List,
        engine_id: str
    ) -> Dict:
        """
        Analyze prediction stability for a single engine
        
        Calculates: |RUL_current - RUL_previous| / time_elapsed_hours
        """
        change_rates = []
        consecutive_preds = []
        
        for i in range(len(predictions)):
            pred = predictions[i]
            
            if i == 0:
                # First prediction
                consecutive_preds.append({
                    'time': pred.prediction_time,
                    'rul': pred.remaining_useful_life,
                    'change_rate': None
                })
            else:
                # Calculate change rate
                prev_pred = predictions[i-1]
                
                rul_delta = abs(pred.remaining_useful_life - prev_pred.remaining_useful_life)
                time_delta = pred.prediction_time - prev_pred.prediction_time
                hours_elapsed = time_delta.total_seconds() / 3600
                
                if hours_elapsed > 0:
                    change_rate = rul_delta / hours_elapsed
                    change_rates.append(change_rate)
                    
                    consecutive_preds.append({
                        'time': pred.prediction_time,
                        'rul': pred.remaining_useful_life,
                        'change_rate': float(change_rate)
                    })
                else:
                    consecutive_preds.append({
                        'time': pred.prediction_time,
                        'rul': pred.remaining_useful_life,
                        'change_rate': None
                    })
        
        if not change_rates:
            return {
                'engine_id': engine_id,
                'status': 'insufficient_data',
                'prediction_count': len(predictions),
                'avg_rul_change_rate': None,
                'max_rul_change_rate': None,
                'consecutive_predictions': consecutive_preds[:10],  # Limit to 10 for response size
                'alert': 'Unable to calculate change rates'
            }
        
        # Calculate metrics
        avg_change_rate = np.mean(change_rates)
        max_change_rate = np.max(change_rates)
        std_change_rate = np.std(change_rates)
        
        # Determine status
        status = 'stable'
        alert = None
        
        if max_change_rate > self.threshold * 2:
            status = 'unstable'
            alert = f"High prediction volatility detected (max rate: {max_change_rate:.2f})"
        elif avg_change_rate > self.threshold:
            status = 'unstable'
            alert = f"Average change rate exceeds threshold (avg: {avg_change_rate:.2f})"
        elif std_change_rate > self.threshold:
            status = 'warning'
            alert = f"High variance in prediction changes (std: {std_change_rate:.2f})"
        
        return {
            'engine_id': engine_id,
            'status': status,
            'prediction_count': len(predictions),
            'avg_rul_change_rate': float(avg_change_rate),
            'max_rul_change_rate': float(max_change_rate),
            'consecutive_predictions': consecutive_preds[:10],  # Limit to 10
            'alert': alert
        }
