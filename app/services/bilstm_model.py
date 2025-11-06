"""
    BiLSTM Model service for binary classification.
"""

from __future__ import annotations

import json
import pathlib
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from app.models.schemas import CmapssDataPoint
from app.utils.dataset import NormalisationStats, build_last_window
from app.core.logging_config import logger

_COL_NAMES = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
_BILSTM_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
_BILSTM_SENSOR_COLS = [f"s{i}" for i in _BILSTM_SENSORS]


def load_config(path: str | pathlib.Path) -> dict:
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"Configuration file {path} is empty")
    return data


class BiLSTMClassifierFixed(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32], fc_sizes=[16, 8], dropout=0.2):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0],
                               num_layers=1, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=hidden_sizes[0]*2, hidden_size=hidden_sizes[1],
                               num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_sizes[1]*2, fc_sizes[0])
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(fc_sizes[1], 1)

    def forward(self, x):
        output, _ = self.bilstm1(x)
        output, _ = self.bilstm2(output)
        last_output = output[:, -1, :]
        x = F.relu(self.fc1(last_output))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        output = self.fc_out(x)
        return output


class BiLSTMPredictionEngine:
    _settings: List[str] = ["setting_1", "setting_2", "setting_3"]

    def __init__(
        self,
        run_dir: str | pathlib.Path,
        *,
        checkpoint: str | pathlib.Path | None = None,
        device: str | torch.device | None = None,
        smoothing_window: int = 5,
        load_checkpoint: bool = True,
    ) -> None:
        self.run_dir = pathlib.Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory '{self.run_dir}' not found")

        logger.info(f"Loading BiLSTM model from {self.run_dir}")
        self.config = load_config(self.run_dir / "config.yaml")
        dataset_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})

        self.stats = self._load_normalisation(self.run_dir / "normalisation.json")
        self.seq_len = dataset_cfg.get("sliding_window", 21)
        self.sensors = list(_BILSTM_SENSOR_COLS)
        self._required_columns = ["unit", "cycle", *self._settings, *self.sensors]
        self.smoothing_window = max(1, smoothing_window)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"Building BiLSTM model on device: {self.device}")
        self.model = BiLSTMClassifierFixed(
            input_size=len(self.sensors),
            hidden_sizes=model_cfg.get("hidden_sizes", [64, 32]),
            fc_sizes=model_cfg.get("fc_sizes", [16, 8]),
            dropout=model_cfg.get("dropout", 0.2),
        ).to(self.device)

        if load_checkpoint:
            ckpt_path = pathlib.Path(checkpoint) if checkpoint else self.run_dir / "checkpoints" / "best_classifier.pt"
            logger.info(f"Loading checkpoint from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info("BiLSTM model loaded successfully")

    def _load_normalisation(self, path: pathlib.Path) -> NormalisationStats:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return NormalisationStats.from_dict(payload)

    def predict_from_api_data(self, engine_id: str, data: List[CmapssDataPoint]) -> int:
        """Return binary classification based on 0.5 probability threshold."""
        prob = self.predict_probability_from_api_data(engine_id, data)
        return int(prob >= 0.5)

    def predict_probability_from_api_data(self, engine_id: str, data: List[CmapssDataPoint]) -> float:
        """Return failure probability (between 0 and 1) for the given engine."""
        df = self._api_data_to_dataframe(engine_id, data)
        processed = self._preprocess_dataframe(df)
        norm_df = self.stats.apply(processed)
        windows, _ = build_last_window(norm_df, self.sensors, self.seq_len)

        tensor = torch.from_numpy(windows).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).squeeze()
            probs = torch.sigmoid(logits)

        return float(probs.item())

    def _api_data_to_dataframe(self, engine_id: str, data: List[CmapssDataPoint]) -> pd.DataFrame:
        rows = [point.model_dump() for point in data]
        df = pd.DataFrame(rows)
        for col in _COL_NAMES:
            if col not in df.columns:
                if col == "unit":
                    try:
                        df["unit"] = int(engine_id.split("-")[-1])
                    except (ValueError, IndexError):
                        df["unit"] = 1
                elif col == "cycle":
                    df["cycle"] = range(len(df))
                else:
                    df[col] = np.nan
        return df[_COL_NAMES]

    def _preprocess_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self._required_columns if col not in raw_df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        df = raw_df[self._required_columns].copy()
        for column in self._required_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["unit", "cycle"], inplace=True)

        if df.empty:
            raise ValueError("No valid rows after cleaning unit/cycle columns")

        df["unit"] = df["unit"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        df.sort_values(["unit", "cycle"], inplace=True)
        df = df.drop_duplicates(subset=["unit", "cycle"], keep="last")

        processed_units: List[pd.DataFrame] = []
        for unit_id, unit_df in df.groupby("unit"):
            unit_processed = self._process_unit(unit_id, unit_df)
            processed_units.append(unit_processed)
        processed = pd.concat(processed_units, ignore_index=True)

        sensor_values = processed[self.sensors].to_numpy(dtype=np.float32)
        sensor_values = self.stats.fill_missing(sensor_values)
        sensor_values = self.stats.clip_outliers(sensor_values)
        sensor_values = self.stats.clip_to_range(sensor_values)
        processed.loc[:, self.sensors] = sensor_values

        if self.smoothing_window > 1:
            processed.sort_values(["unit", "cycle"], inplace=True)
            smoothed = processed.groupby("unit")[self.sensors].transform(
                lambda s: s.rolling(self.smoothing_window, min_periods=1).median()
            )
            smoothed_values = smoothed.to_numpy(dtype=np.float32)
            smoothed_values = self.stats.fill_missing(smoothed_values)
            smoothed_values = self.stats.clip_outliers(smoothed_values)
            smoothed_values = self.stats.clip_to_range(smoothed_values)
            processed.loc[:, self.sensors] = smoothed_values

        return processed

    def _process_unit(self, unit_id: int, unit_df: pd.DataFrame) -> pd.DataFrame:
        unit_df = unit_df.set_index("cycle")
        full_index = np.arange(unit_df.index.min(), unit_df.index.max() + 1)
        unit_df = unit_df.reindex(full_index)
        unit_df["unit"] = unit_id

        unit_df[self._settings] = unit_df[self._settings].interpolate(method="linear", limit_direction="both")
        unit_df[self._settings] = unit_df[self._settings].fillna(0.0)

        unit_df[self.sensors] = unit_df[self.sensors].interpolate(method="linear", limit_direction="both")
        unit_df[self.sensors] = unit_df[self.sensors].fillna(unit_df[self.sensors].median())

        unit_df.reset_index(inplace=True)
        unit_df.rename(columns={"index": "cycle"}, inplace=True)
        return unit_df
