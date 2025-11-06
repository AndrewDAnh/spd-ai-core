"""Retraining routines shared by the training job manager."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.logging_config import logger
from app.models.schemas import RetrainingDataset
from app.services.bilstm_model import BiLSTMPredictionEngine
from app.services.star_model import STARPredictionEngine
from app.utils.dataset import build_inference_windows, build_last_window


class _EarlyStopping:
    """Simple patience-based early stopping helper."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.counter = 0
        self.best_state: Dict[str, torch.Tensor] | None = None

    def update(self, loss: float, model: nn.Module) -> bool:
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.best_state = deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def apply(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _prepare_regression_windows(
    dataset: RetrainingDataset,
    engine: STARPredictionEngine,
) -> Tuple[np.ndarray, np.ndarray]:
    windows: List[np.ndarray] = []
    targets: List[float] = []

    samples = dataset.regression_samples or []
    for sample in samples:
        if not sample.data:
            continue
        df = engine._api_data_to_dataframe(sample.engine_id, sample.data)  # type: ignore[attr-defined]
        processed = engine._preprocess_dataframe(df)  # type: ignore[attr-defined]
        norm_df = engine.stats.apply(processed)
        seq_windows, _ = build_inference_windows(norm_df, engine.sensors, engine.seq_len)
        if len(seq_windows) == 0:
            continue
        windows.append(seq_windows[-1])
        targets.append(float(sample.target_rul))

    if not windows:
        raise ValueError("No regression samples provided for retraining")

    return np.stack(windows).astype(np.float32), np.asarray(targets, dtype=np.float32)


def _prepare_classification_windows(
    dataset: RetrainingDataset,
    engine: BiLSTMPredictionEngine,
) -> Tuple[np.ndarray, np.ndarray]:
    windows: List[np.ndarray] = []
    labels: List[int] = []

    samples = dataset.classification_samples or []
    for sample in samples:
        if not sample.data:
            continue
        df = engine._api_data_to_dataframe(sample.engine_id, sample.data)
        processed = engine._preprocess_dataframe(df)
        norm_df = engine.stats.apply(processed)
        seq_windows, _ = build_last_window(norm_df, engine.sensors, engine.seq_len)
        if len(seq_windows) == 0:
            continue
        windows.append(seq_windows[-1])
        labels.append(int(sample.label))

    if not windows:
        raise ValueError("No classification samples provided for retraining")

    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.float32)


def _split_dataset(tensors: Tuple[torch.Tensor, torch.Tensor], val_ratio: float = 0.2) -> Tuple:
    data, targets = tensors
    total = data.size(0)
    if total < 2:
        return (TensorDataset(data, targets), None)

    val_size = int(max(1, total * val_ratio)) if total >= 5 else 0
    if val_size == 0:
        return (TensorDataset(data, targets), None)

    indices = torch.randperm(total)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if len(train_indices) == 0:
        train_indices = val_indices
        val_indices = val_indices[:0]

    train_dataset = TensorDataset(data[train_indices], targets[train_indices])
    val_dataset = TensorDataset(data[val_indices], targets[val_indices]) if len(val_indices) else None
    return train_dataset, val_dataset


def _run_regression_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> float:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = _EarlyStopping(patience=3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for windows, target in train_loader:
            windows = windows.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            preds = model(windows)
            loss = criterion(preds, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item() * windows.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        if val_loader is None:
            should_stop = stopper.update(train_loss, model)
        else:
            val_loss = _evaluate_regression(model, val_loader, device, criterion)
            should_stop = stopper.update(val_loss, model)
        logger.info("Regression epoch %d: train_loss=%.4f", epoch + 1, train_loss)
        if should_stop:
            logger.info("Early stopping triggered for regression model")
            break

    stopper.apply(model)
    model.eval()
    return stopper.best_loss


def _evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for windows, target in loader:
            windows = windows.to(device)
            target = target.to(device)
            preds = model(windows)
            loss = criterion(preds, target)
            total_loss += loss.item() * windows.size(0)
    return total_loss / len(loader.dataset)


def _run_classification_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> float:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = _EarlyStopping(patience=3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for windows, labels in train_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(windows).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * windows.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        if val_loader is None:
            should_stop = stopper.update(train_loss, model)
        else:
            val_loss = _evaluate_classification(model, val_loader, device, criterion)
            should_stop = stopper.update(val_loss, model)
        logger.info("Classification epoch %d: train_loss=%.4f", epoch + 1, train_loss)
        if should_stop:
            logger.info("Early stopping triggered for classification model")
            break

    stopper.apply(model)
    model.eval()
    return stopper.best_loss


def _evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for windows, labels in loader:
            windows = windows.to(device)
            labels = labels.to(device)
            logits = model(windows).squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * windows.size(0)
    return total_loss / len(loader.dataset)


def train_regression_model(
    dataset: RetrainingDataset,
    *,
    run_dir: str,
    device: str,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Fine-tune the STAR regression model on the supplied dataset."""

    artifact_dir = Path(run_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    engine = STARPredictionEngine(run_dir=artifact_dir, device=device, load_checkpoint=False)
    model = engine.model
    model.train()
    torch_device = torch.device(device)

    windows_np, targets_np = _prepare_regression_windows(dataset, engine)

    windows = torch.from_numpy(windows_np)
    targets = torch.from_numpy(targets_np)
    train_dataset, val_dataset = _split_dataset((windows, targets), val_ratio=0.2)

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
        if val_dataset is not None
        else None
    )

    best_loss = _run_regression_training(
        model,
        train_loader,
        val_loader,
        torch_device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )

    model.eval()
    with torch.no_grad():
        preds = model(windows.to(torch_device)).cpu().numpy()
    targets_array = targets_np

    mae = mean_absolute_error(targets_array, preds)
    mse = mean_squared_error(targets_array, preds)
    mape = float(np.mean(np.abs((targets_array - preds) / np.maximum(targets_array, 1e-6))))

    checkpoints_dir = artifact_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoints_dir / "best.pt")

    logger.info(
        "Regression retraining complete: samples=%d, mse=%.4f, mae=%.4f, mape=%.4f",
        len(windows_np),
        mse,
        mae,
        mape,
    )

    return {
        "loss": float(best_loss),
        "mse": float(mse),
        "mae": float(mae),
        "mape": float(mape),
        "samples": float(len(windows_np)),
    }


def train_classification_model(
    dataset: RetrainingDataset,
    *,
    run_dir: str,
    device: str,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
) -> Dict[str, float]:
    """Fine-tune the BiLSTM classification model on the supplied dataset."""

    artifact_dir = Path(run_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    engine = BiLSTMPredictionEngine(run_dir=artifact_dir, device=device, load_checkpoint=False)
    model = engine.model
    model.train()
    torch_device = torch.device(device)

    windows_np, labels_np = _prepare_classification_windows(dataset, engine)

    windows = torch.from_numpy(windows_np)
    labels = torch.from_numpy(labels_np)
    train_dataset, val_dataset = _split_dataset((windows, labels), val_ratio=0.2)

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
        if val_dataset is not None
        else None
    )

    best_loss = _run_classification_training(
        model,
        train_loader,
        val_loader,
        torch_device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )

    model.eval()
    with torch.no_grad():
        logits = model(windows.to(torch_device)).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    precision, recall, f1_scores, _ = precision_recall_fscore_support(
        labels_np,
        preds,
        labels=[0, 1],
        zero_division=0.0,
    )
    macro_f1 = float(np.mean(f1_scores))

    checkpoints_dir = artifact_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoints_dir / "best_classifier.pt")

    logger.info(
        "Classification retraining complete: samples=%d, f1=%.4f",
        len(windows_np),
        macro_f1,
    )

    return {
        "loss": float(best_loss),
        "precision_0": float(precision[0]),
        "precision_1": float(precision[1]),
        "recall_0": float(recall[0]),
        "recall_1": float(recall[1]),
        "f1_macro": macro_f1,
        "samples": float(len(windows_np)),
    }
