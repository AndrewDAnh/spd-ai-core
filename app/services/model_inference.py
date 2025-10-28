import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime
from app.core.config import get_settings
from app.core.logging_config import logger
from app.models.schemas import CmapssDataPoint
from app.services.star_model import STARPredictionEngine


class ModelInferenceService:
    """Service for model inference using STAR regression model"""
    
    def __init__(self):
        self.settings = get_settings()
        self.regression_model: Optional[STARPredictionEngine] = None
        self.classification_model = None  # Placeholder for future classifier
        self._load_models()
    
    def _load_models(self):
        """Load trained models"""
        try:
            logger.info("Loading STAR regression model...")
            self.regression_model = STARPredictionEngine(
                run_dir=self.settings.REGRESSION_MODEL_PATH,
                device=self.settings.DEVICE,
                smoothing_window=5
            )
            logger.info("STAR regression model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load regression model: {str(e)}")
            logger.warning("Model inference will not be available")
            raise
        
        # Classification model placeholder
        logger.info("Classification model not available - will return None for is_going_to_fail")
    
    def predict(
        self,
        engine_id: str,
        data: List[CmapssDataPoint],
        timestamp: datetime
    ) -> Tuple[float, Optional[bool], float]:
        """
        Make RUL prediction for an engine
        
        Args:
            engine_id: Engine identifier
            data: Time-series sensor data in C-MAPSS format
            timestamp: Prediction timestamp
            
        Returns:
            Tuple of (remaining_useful_life, is_going_to_fail, confidence)
        """
        try:
            if self.regression_model is None:
                raise RuntimeError("Regression model not loaded")
            
            # Get RUL prediction from STAR model
            rul = self.regression_model.predict_from_api_data(engine_id, data)
            
            # Classification placeholder - return None when classifier unavailable
            is_going_to_fail = None
            
            # TODO: When classifier is available:
            # is_going_to_fail = self.classification_model.predict(...)
            
            # Calculate confidence based on RUL value
            confidence = self._calculate_confidence(rul)
            
            logger.info(f"Prediction for {engine_id}: RUL={rul:.2f}, confidence={confidence:.2f}")
            return float(rul), is_going_to_fail, float(confidence)
            
        except Exception as e:
            logger.error(f"Error in prediction for engine {engine_id}: {str(e)}")
            raise
    
    def _calculate_confidence(self, rul: float) -> float:
        """
        Calculate prediction confidence based on RUL value
        
        Uses a heuristic based on the RUL range.
        In the future, this could be replaced with model uncertainty estimates.
        """
        # Higher confidence for very low RUL (imminent failure) or very high RUL (healthy)
        # Lower confidence for middle range where degradation patterns are less clear
        if rul < 30:
            # Very low RUL - high confidence in critical state
            return np.clip(0.90 + np.random.uniform(-0.05, 0.05), 0.85, 0.95)
        elif rul > 100:
            # High RUL - high confidence in healthy state
            return np.clip(0.85 + np.random.uniform(-0.05, 0.05), 0.80, 0.90)
        else:
            # Middle range - moderate confidence
            return np.clip(0.75 + np.random.uniform(-0.05, 0.05), 0.70, 0.80)

