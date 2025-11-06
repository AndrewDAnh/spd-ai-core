"""Data preparation utilities for the C-MAPSS turbofan datasets."""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_DEFAULT_SENSORS = [
    2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21,
]

_COL_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"s{i}" for i in range(1, 22)]


@dataclass
class NormalisationStats:
    sensors: List[str]
    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    median: np.ndarray
    mad: np.ndarray

    def to_dict(self) -> dict:
        return {
            "sensors": self.sensors,
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "median": self.median.tolist(),
            "mad": self.mad.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NormalisationStats":
        sensors = list(payload["sensors"])
        minimum = np.asarray(payload["minimum"], dtype=np.float32)
        maximum = np.asarray(payload["maximum"], dtype=np.float32)
        mean = np.asarray(payload.get("mean", minimum), dtype=np.float32)
        std = np.asarray(payload.get("std", np.ones_like(minimum)), dtype=np.float32)
        median = np.asarray(payload.get("median", minimum), dtype=np.float32)
        mad = np.asarray(payload.get("mad", np.ones_like(minimum)), dtype=np.float32)
        return cls(sensors, minimum, maximum, mean, std, median, mad)

    def apply(self, values: pd.DataFrame) -> pd.DataFrame:
        result = values.copy()
        array = result[self.sensors].to_numpy(dtype=np.float32)
        array = self.clip_outliers(array)
        array = self.clip_to_range(array)
        denom = np.maximum(self.maximum - self.minimum, 1e-6)
        normed = (array - self.minimum) / denom
        result[self.sensors] = normed
        return result

    def clip_outliers(self, values: np.ndarray, sigma: float = 6.0) -> np.ndarray:
        eps = 1e-6
        std = np.maximum(self.std, eps)
        mad = np.maximum(self.mad, std)
        bandwidth = np.minimum(sigma * std, 3.5 * mad)
        lower = (self.mean - bandwidth)[None, :]
        upper = (self.mean + bandwidth)[None, :]
        return np.clip(values, lower, upper)

    def clip_to_range(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.minimum[None, :], self.maximum[None, :])

    def fill_missing(self, values: np.ndarray) -> np.ndarray:
        if not np.isnan(values).any():
            return values
        row_idx, col_idx = np.where(np.isnan(values))
        values[row_idx, col_idx] = self.median[col_idx]
        return values


class SequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Simple Tensor dataset returning (window, target) pairs."""

    def __init__(self, windows: np.ndarray, targets: np.ndarray):
        self.windows = windows.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = torch.from_numpy(self.windows[index])
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return window, target


class EvaluationDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Dataset returning (window, target, unit_id) for the official test split."""

    def __init__(self, windows: np.ndarray, targets: np.ndarray, unit_ids: np.ndarray):
        self.windows = windows.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.unit_ids = unit_ids.astype(np.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = torch.from_numpy(self.windows[index])
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        unit = torch.tensor(self.unit_ids[index], dtype=torch.int64)
        return window, target, unit


class CmapssDataModule:
    """Prepare sliding window datasets for STAR training and evaluation."""

    def __init__(
        self,
        root: str | pathlib.Path,
        dataset: str,
        seq_len: int,
        stride: int,
        sensors: Iterable[int] | None,
        rul_max: int,
        val_split: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        self.root = pathlib.Path(root)
        self.dataset = dataset.upper()
        self.seq_len = seq_len
        self.stride = stride
        self.sensor_ids = list(sensors) if sensors else list(_DEFAULT_SENSORS)
        self.rul_max = rul_max
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.stats: NormalisationStats | None = None
        self.train_ds: SequenceDataset | None = None
        self.val_ds: SequenceDataset | None = None
        self.test_ds: EvaluationDataset | None = None

    @property
    def sensor_names(self) -> List[str]:
        return [f"s{i}" for i in self.sensor_ids]

    def prepare(self) -> None:
        train_path = self.root / f"train_{self.dataset}.txt"
        test_path = self.root / f"test_{self.dataset}.txt"
        rul_path = self.root / f"RUL_{self.dataset}.txt"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing training file: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file: {test_path}")
        if not rul_path.exists():
            raise FileNotFoundError(f"Missing RUL file: {rul_path}")

        train_df = _read_raw(train_path)
        test_df = _read_raw(test_path)
        rul_df = pd.read_csv(rul_path, header=None, names=["rul"])  # column of final RULs

        units = sorted(train_df["unit"].unique())
        val_count = max(1, int(len(units) * self.val_split))
        val_units = set(units[-val_count:])
        train_units = set(units[:-val_count])
        if not train_units:
            train_units = set(units)
        if not val_units:
            val_units = set(units)

        stats = _compute_normalisation(train_df, train_units, self.sensor_names)
        self.stats = stats

        train_df = stats.apply(train_df)
        test_df = stats.apply(test_df)

        train_df["rul"] = _compute_truncated_rul(train_df, self.rul_max)
        train_windows, train_targets = _build_windows(
            train_df, train_units, self.sensor_names, self.seq_len, self.stride
        )
        val_windows, val_targets = _build_windows(
            train_df, val_units, self.sensor_names, self.seq_len, self.stride
        )

        test_windows, test_targets, test_units = _build_test_windows(
            test_df,
            rul_df,
            self.sensor_names,
            self.seq_len,
            self.rul_max,
        )

        self.train_ds = SequenceDataset(train_windows, train_targets)
        self.val_ds = SequenceDataset(val_windows, val_targets)
        self.test_ds = EvaluationDataset(test_windows, test_targets, test_units)

    def train_dataloader(self) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
        if self.train_ds is None:
            raise RuntimeError("Data module not prepared")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
        if self.val_ds is None:
            raise RuntimeError("Data module not prepared")
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.test_ds is None:
            raise RuntimeError("Data module not prepared")
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def _read_raw(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, delim_whitespace=True, header=None, names=_COL_NAMES)


def _compute_normalisation(
    df: pd.DataFrame, train_units: set[int], sensors: Sequence[str]
) -> NormalisationStats:
    subset = df[df["unit"].isin(train_units)]
    sensor_df = subset[sensors]
    minimum = sensor_df.min().to_numpy(dtype=np.float32)
    maximum = sensor_df.max().to_numpy(dtype=np.float32)
    mean = sensor_df.mean().to_numpy(dtype=np.float32)
    std = sensor_df.std(ddof=0).replace(0, np.nan).to_numpy(dtype=np.float32)
    std = np.nan_to_num(std, nan=1.0)
    median = sensor_df.median().to_numpy(dtype=np.float32)
    mad = (sensor_df.subtract(median)).abs().median().to_numpy(dtype=np.float32)
    mad = np.where(mad < 1e-6, std, mad)
    return NormalisationStats(list(sensors), minimum, maximum, mean, std, median, mad)


def _compute_truncated_rul(df: pd.DataFrame, rul_max: int) -> pd.Series:
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycles - df["cycle"]).clip(lower=0)
    return np.minimum(rul, rul_max)


def _build_windows(
    df: pd.DataFrame,
    units: set[int],
    sensors: Sequence[str],
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    windows: List[np.ndarray] = []
    targets: List[float] = []
    for unit in sorted(units):
        unit_df = df[df["unit"] == unit]
        values = unit_df[sensors].to_numpy(dtype=np.float32)
        rul = unit_df["rul"].to_numpy(dtype=np.float32)
        if len(values) < seq_len:
            continue
        for end in range(seq_len, len(values) + 1, stride):
            start = end - seq_len
            window = values[start:end]
            target = rul[end - 1]
            windows.append(window)
            targets.append(target)
    if not windows:
        raise ValueError(
            "No training/validation windows generated. Review seq_len, stride, or unit split."
        )
    return np.stack(windows), np.array(targets)


def _build_test_windows(
    df: pd.DataFrame,
    rul_df: pd.DataFrame,
    sensors: Sequence[str],
    seq_len: int,
    rul_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: List[np.ndarray] = []
    targets: List[float] = []
    unit_ids: List[int] = []
    units = sorted(df["unit"].unique())
    if len(units) != len(rul_df):
        raise ValueError("RUL file does not match number of test units")
    for idx, unit in enumerate(units):
        unit_df = df[df["unit"] == unit]
        values = unit_df[sensors].to_numpy(dtype=np.float32)
        if len(values) >= seq_len:
            window = values[-seq_len:]
        else:
            pad = np.repeat(values[:1], seq_len - len(values), axis=0)
            window = np.concatenate([pad, values], axis=0)
        rul_target = float(min(rul_df.iloc[idx, 0], rul_max))
        windows.append(window)
        targets.append(rul_target)
        unit_ids.append(unit)
    if not windows:
        raise ValueError("No test windows generated")
    return np.stack(windows), np.array(targets), np.array(unit_ids)


def build_inference_windows(
    df: pd.DataFrame,
    sensors: Sequence[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate model-ready windows from a raw CMAPSS-style dataframe.

    Returns
    -------
    windows: np.ndarray
        Array shaped (num_units, seq_len, num_sensors).
    unit_ids: np.ndarray
        Engine identifiers aligned with ``windows``.
    """
    windows: List[np.ndarray] = []
    unit_ids: List[int] = []
    units = sorted(df["unit"].unique())
    if not units:
        raise ValueError("Input dataframe contains no engines")
    for unit in units:
        unit_df = df[df["unit"] == unit]
        values = unit_df[sensors].to_numpy(dtype=np.float32)
        if len(values) >= seq_len:
            window = values[-seq_len:]
        else:
            pad = np.repeat(values[:1], seq_len - len(values), axis=0)
            window = np.concatenate([pad, values], axis=0)
        windows.append(window)
        unit_ids.append(unit)
    return np.stack(windows), np.array(unit_ids)

def build_last_window(
    df: pd.DataFrame,
    sensors: Sequence[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the last window per unit from a CMAPSS-style dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns ['unit', 'cycle', ...sensor columns...]
    sensors : Sequence[str]
        List of sensor column names to use.
    seq_len : int
        Length of the window (number of cycles).

    Returns
    -------
    windows : np.ndarray
        Array shaped (num_units, seq_len, num_sensors).
    unit_ids : np.ndarray
        Engine identifiers aligned with ``windows``.
    """
    windows: List[np.ndarray] = []
    unit_ids: List[int] = []
    units = sorted(df["unit"].unique())
    if not units:
        raise ValueError("Input dataframe contains no engines")

    for unit in units:
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        values = unit_df[sensors].to_numpy(dtype=np.float32)
        if len(values) >= seq_len:
            window = values[-seq_len:]
        else:
            pad = np.repeat(values[:1], seq_len - len(values), axis=0)
            window = np.concatenate([pad, values], axis=0)
        windows.append(window)
        unit_ids.append(unit)

    return np.stack(windows), np.array(unit_ids)
