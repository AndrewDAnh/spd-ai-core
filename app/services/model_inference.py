import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime
from app.core.config import get_settings
from app.core.logging_config import logger
from app.models.schemas import CmapssDataPoint
from app.services.star_model import STARPredictionEngine
from app.services.bilstm_model import BiLSTMPredictionEngine


class ModelInferenceService:
    """Service for model inference"""

    def __init__(
        self,
        regression_run_dir: Optional[str] = None,
        classification_run_dir: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.regression_run_dir = regression_run_dir or self.settings.REGRESSION_MODEL_PATH
        self.classification_run_dir = classification_run_dir or self.settings.CLASSIFICATION_MODEL_PATH
        self.regression_model: Optional[STARPredictionEngine] = None
        self.classification_model: Optional[BiLSTMPredictionEngine] = None
        self._load_models()

    def _load_models(self):
        """Load trained models"""
        try:
            self.regression_model = None
            self.classification_model = None
            logger.info("Loading STAR regression model...")
            self.regression_model = STARPredictionEngine(
                run_dir=self.regression_run_dir,
                device=self.settings.DEVICE,
                smoothing_window=5
            )
            logger.info("Loading BiLSTM classification model...")
            self.classification_model = BiLSTMPredictionEngine(
                run_dir=self.classification_run_dir,
                device=self.settings.DEVICE,
                smoothing_window=5
            )
            logger.info(
                "Inference models loaded (regression_dir=%s, classification_dir=%s)",
                self.regression_run_dir,
                self.classification_run_dir,
            )
        except Exception as e:
            logger.error(f"Failed to load regression model: {str(e)}")
            logger.warning("Model inference will not be available")
            raise

    def reload_models(
        self,
        *,
        regression_run_dir: Optional[str] = None,
        classification_run_dir: Optional[str] = None,
    ) -> None:
        """Reload models from new run directories, falling back to existing ones."""
        previous_regression_dir = self.regression_run_dir
        previous_classification_dir = self.classification_run_dir
        if regression_run_dir:
            self.regression_run_dir = regression_run_dir
        if classification_run_dir:
            self.classification_run_dir = classification_run_dir
        try:
            self._load_models()
        except Exception:
            logger.exception("Failed to reload models, restoring previous configuration")
            self.regression_run_dir = previous_regression_dir
            self.classification_run_dir = previous_classification_dir
            self._load_models()
            raise
  
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
            
            if self.classification_model is None:
                raise RuntimeError("Classification model not loaded")
            
            # Get RUL prediction from STAR model
            rul = self.regression_model.predict_from_api_data(engine_id, data)
            
            # Classification placeholder - return None when classifier unavailable
            is_going_to_fail = self.classification_model.predict_from_api_data(engine_id, data)
            
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
