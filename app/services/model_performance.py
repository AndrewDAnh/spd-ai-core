"""Service layer for evaluating model performance on the official C-MAPSS test split."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from app.core.config import get_settings
from app.core.logging_config import logger
from app.models.schemas import CmapssDataPoint
from app.services.model_inference import ModelInferenceService
from app.utils.cmapss_loader import load_test_dataset, load_rul_values


@dataclass
class ModelPerformanceResult:
    """Container for regression and placeholder classification metrics."""

    mean_squared_error: Optional[float]
    mean_absolute_error: Optional[float]
    mean_absolute_percentage_error: Optional[float]
    precision: Optional[List[float]]
    recall: Optional[List[float]]
    f1_score: float
    validation_time: datetime

    def to_dict(self) -> Dict[str, object]:
        return {
            "mean_squared_error": self.mean_squared_error,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "validation_time": self.validation_time,
        }


class ModelPerformanceService:
    """Evaluate STAR model predictions against the held-out FD001 test set."""

    def __init__(self, model_service: ModelInferenceService | None = None) -> None:
        self.settings = get_settings()
        self.model_service = model_service or ModelInferenceService()

    def run_evaluation(self) -> ModelPerformanceResult:
        """Run inference on the test split and compute regression metrics."""
        test_df = load_test_dataset(self._dataset_root)
        rul_df = load_rul_values(self._dataset_root)

        engine_ids = sorted(test_df["unit"].unique().tolist())
        if len(engine_ids) != len(rul_df):
            raise ValueError(
                "Mismatch between test engines and RUL labels. "
                "Ensure test_FD001.txt and RUL_FD001.txt correspond to the same split."
            )

        predictions: List[float] = []
        targets: List[float] = []

        for idx, unit_id in enumerate(engine_ids):
            engine_df = test_df[test_df["unit"] == unit_id]
            datapoints = [self._row_to_datapoint(row) for _, row in engine_df.iterrows()]
            engine_key = f"ENG-{int(unit_id):03d}"
            predicted_rul, _, _ = self.model_service.predict(
                engine_id=engine_key,
                data=datapoints,
                timestamp=datetime.utcnow(),
            )
            predictions.append(float(predicted_rul))
            targets.append(float(rul_df.iloc[idx, 0]))

        y_true = np.asarray(targets, dtype=np.float32)
        y_pred = np.asarray(predictions, dtype=np.float32)

        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(mean_absolute_percentage_error(y_true, y_pred))

        validation_time = datetime.utcnow()

        # Classification placeholders until classifier is available.
        precision: List[float] = []
        recall: List[float] = []
        f1_score: float = 0.0

        return ModelPerformanceResult(
            mean_squared_error=mse,
            mean_absolute_error=mae,
            mean_absolute_percentage_error=mape,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            validation_time=validation_time,
        )

    @property
    def _dataset_root(self) -> str:
        # The settings expose full file paths; we only need the directory.
        dataset_parent = Path(self.settings.TEST_DATASET_PATH).resolve().parent
        rul_parent = Path(self.settings.TEST_RUL_PATH).resolve().parent
        if dataset_parent != rul_parent:
            logger.warning(
                "Dataset and RUL files reside in different directories. Using %s as root.",
                dataset_parent,
            )
        return str(dataset_parent)

    def _row_to_datapoint(self, row: pd.Series) -> CmapssDataPoint:
        payload: Dict[str, float] = {
            "cycle": int(row["cycle"]),
            "setting_1": float(row["setting_1"]),
            "setting_2": float(row["setting_2"]),
            "setting_3": float(row["setting_3"]),
        }
        for sensor_idx in range(1, 22):
            sensor_key = f"s{sensor_idx}"
            payload[sensor_key] = float(row[sensor_key])
        return CmapssDataPoint(**payload)


# Module-level singleton used by the API routers.
model_performance_service = ModelPerformanceService()
