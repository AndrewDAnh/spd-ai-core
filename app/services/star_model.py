"""STAR model service for RUL prediction.

Adapted from regression_pipeline_ref/inference_regression.py for API integration.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from app.models.schemas import CmapssDataPoint
from app.utils.dataset import NormalisationStats, build_inference_windows
from app.core.logging_config import logger

_COL_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"s{i}" for i in range(1, 22)]


def load_config(path: str | pathlib.Path) -> Dict[str, object]:
    """Load a YAML configuration file into a dictionary."""
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ValueError(f"Configuration file {path} is empty")
    return data


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchEmbedding(nn.Module):
    def __init__(self, num_sensors: int, seq_len: int, patch_len: int, dim: int, dropout: float) -> None:
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len")
        self.num_sensors = num_sensors
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.proj = nn.Linear(patch_len, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_sensors, self.num_patches, dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, sensors = x.shape
        if sensors != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensors, received {sensors}")
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch, sensors, self.num_patches, self.patch_len)
        x = self.proj(x)
        x = x + self.pos_embed
        return self.dropout(x)


class TwoStageAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_norm1 = nn.LayerNorm(dim)
        self.temporal_norm2 = nn.LayerNorm(dim)
        self.temporal_ff = FeedForward(dim, hidden_dim, dropout)
        self.temporal_dropout = nn.Dropout(dropout)

        self.sensor_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.sensor_norm1 = nn.LayerNorm(dim)
        self.sensor_norm2 = nn.LayerNorm(dim)
        self.sensor_ff = FeedForward(dim, hidden_dim, dropout)
        self.sensor_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, sensors, patches, dim = x.shape
        temp_in = x.reshape(b * sensors, patches, dim)
        temp_out, _ = self.temporal_attn(temp_in, temp_in, temp_in)
        temp_in = self.temporal_norm1(temp_in + self.temporal_dropout(temp_out))
        temp_ff = self.temporal_ff(temp_in)
        temp_in = self.temporal_norm2(temp_in + self.temporal_dropout(temp_ff))
        x = temp_in.reshape(b, sensors, patches, dim)

        sens_in = x.permute(0, 2, 1, 3).contiguous().reshape(b * patches, sensors, dim)
        sens_out, _ = self.sensor_attn(sens_in, sens_in, sens_in)
        sens_in = self.sensor_norm1(sens_in + self.sensor_dropout(sens_out))
        sens_ff = self.sensor_ff(sens_in)
        sens_in = self.sensor_norm2(sens_in + self.sensor_dropout(sens_ff))
        x = sens_in.reshape(b, patches, sensors, dim).permute(0, 2, 1, 3).contiguous()
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, sensors, patches, dim = x.shape
        if patches % 2 == 1:
            pad = x[:, :, -1:, :]
            x = torch.cat([x, pad], dim=2)
            patches += 1
        x = x.reshape(b, sensors, patches // 2, 2, dim)
        x = x.reshape(b, sensors, patches // 2, 2 * dim)
        return self.reduction(x)


class PatchExpansion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, sensors, patches, dim = x.shape
        x = self.expand(x)
        x = x.reshape(b, sensors, patches, 2, dim)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(b, sensors, patches * 2, dim)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_block = TwoStageAttentionBlock(dim, heads, hidden_dim, dropout)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_norm1 = nn.LayerNorm(dim)
        self.cross_norm2 = nn.LayerNorm(dim)
        self.cross_ff = FeedForward(dim, hidden_dim, dropout)
        self.cross_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.self_block(x)
        b, sensors, patches, dim = x.shape
        query = x.reshape(b, sensors * patches, dim)
        key = memory.reshape(b, memory.shape[1] * memory.shape[2], dim)
        attn_out, _ = self.cross_attn(query, key, key)
        query = self.cross_norm1(query + self.cross_dropout(attn_out))
        ff = self.cross_ff(query)
        query = self.cross_norm2(query + self.cross_dropout(ff))
        return query.reshape(b, sensors, patches, dim)


class EncoderStage(nn.Module):
    def __init__(self, depth: int, dim: int, heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TwoStageAttentionBlock(dim, heads, hidden_dim, dropout) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, depth: int, dim: int, heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(dim, heads, hidden_dim, dropout) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, memory)
        return x


class STARModel(nn.Module):
    def __init__(
        self,
        *,
        num_sensors: int,
        seq_len: int,
        patch_len: int,
        d_model: int,
        num_scales: int,
        encoder_depths: Sequence[int],
        decoder_depths: Sequence[int],
        heads: Sequence[int],
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(encoder_depths) != num_scales or len(decoder_depths) != num_scales:
            raise ValueError("encoder_depths and decoder_depths must match num_scales")
        if len(heads) != num_scales:
            raise ValueError("heads list length must equal num_scales")
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len")

        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_scales = num_scales
        self.patch_counts = [seq_len // patch_len // (2 ** i) for i in range(num_scales)]
        if self.patch_counts[-1] < 1:
            raise ValueError("Too many scales for the chosen seq_len and patch_len")

        hidden_dim = int(d_model * mlp_ratio)
        self.patch_embed = PatchEmbedding(num_sensors, seq_len, patch_len, d_model, dropout)

        self.encoder_stages = nn.ModuleList([
            EncoderStage(depth=encoder_depths[i], dim=d_model, heads=heads[i], hidden_dim=hidden_dim, dropout=dropout)
            for i in range(num_scales)
        ])
        self.patch_mergers = nn.ModuleList([
            PatchMerging(d_model) for _ in range(num_scales - 1)
        ])

        self.decoder_stages = nn.ModuleList([
            DecoderStage(depth=decoder_depths[i], dim=d_model, heads=heads[i], hidden_dim=hidden_dim, dropout=dropout)
            for i in range(num_scales)
        ])
        self.patch_expanders = nn.ModuleList([
            PatchExpansion(d_model) if i < num_scales - 1 else nn.Identity()
            for i in range(num_scales)
        ])
        self.decoder_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_sensors, self.patch_counts[i], d_model))
            for i in range(num_scales)
        ])
        for param in self.decoder_queries:
            nn.init.trunc_normal_(param, std=0.02)

        self.scale_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_scales)
        ])
        self.final_head = nn.Sequential(
            nn.LayerNorm(d_model * num_scales),
            nn.Linear(d_model * num_scales, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        encoder_outputs: List[torch.Tensor] = []
        features = x
        for idx, stage in enumerate(self.encoder_stages):
            features = stage(features)
            encoder_outputs.append(features)
            if idx < self.num_scales - 1:
                features = self.patch_mergers[idx](features)

        batch_size = x.shape[0]
        dec_features: torch.Tensor | None = None
        decoded_per_scale: List[torch.Tensor | None] = [None] * self.num_scales
        for scale in reversed(range(self.num_scales)):
            memory = encoder_outputs[scale]
            query = self.decoder_queries[scale].expand(batch_size, -1, -1, -1)
            if dec_features is None:
                dec_features = query
            else:
                dec_features = self.patch_expanders[scale](dec_features)
                dec_features = dec_features + query
            dec_features = self.decoder_stages[scale](dec_features, memory)
            decoded_per_scale[scale] = dec_features

        scale_embeddings: List[torch.Tensor] = []
        for idx, feat in enumerate(decoded_per_scale):
            if feat is None:
                raise RuntimeError("Decoder failed to produce features for every scale")
            pooled = feat.mean(dim=(1, 2))
            scale_embeddings.append(self.scale_heads[idx](pooled))
        fused = torch.cat(scale_embeddings, dim=-1)
        return self.final_head(fused).squeeze(-1)


class STARPredictionEngine:
    """Prediction engine for STAR model adapted for API use."""
    
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

        logger.info(f"Loading STAR model from {self.run_dir}")
        self.config = load_config(self.run_dir / "config.yaml")
        dataset_cfg = self.config["data"]
        model_cfg = self.config["model"]

        self.stats = self._load_normalisation(self.run_dir / "normalisation.json")
        self.seq_len = dataset_cfg["seq_len"]
        self.sensors = list(self.stats.sensors)
        self._required_columns = ["unit", "cycle", *self._settings, *self.sensors]
        self.smoothing_window = max(1, smoothing_window)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"Building STAR model on device: {self.device}")
        self.model = STARModel(
            num_sensors=len(self.sensors),
            seq_len=dataset_cfg["seq_len"],
            patch_len=dataset_cfg["patch_len"],
            d_model=model_cfg["d_model"],
            num_scales=model_cfg["num_scales"],
            encoder_depths=model_cfg["encoder_depths"],
            decoder_depths=model_cfg["decoder_depths"],
            heads=model_cfg["heads"],
            mlp_ratio=model_cfg.get("mlp_ratio", 2.0),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(self.device)
        
        if load_checkpoint:
            ckpt_path = pathlib.Path(checkpoint) if checkpoint else self.run_dir / "checkpoints" / "best.pt"
            logger.info(f"Loading checkpoint from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            if "model" in state:
                state = state["model"]
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info("STAR model loaded successfully")

    def _load_normalisation(self, path: pathlib.Path) -> NormalisationStats:
        """Load normalisation statistics from JSON file."""
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return NormalisationStats.from_dict(payload)

    def predict_from_api_data(self, engine_id: str, data: List[CmapssDataPoint]) -> float:
        """Make prediction from API data format.
        
        Args:
            engine_id: Engine identifier
            data: List of CmapssDataPoint objects
            
        Returns:
            Predicted RUL value
        """
        # Convert API data to DataFrame
        df = self._api_data_to_dataframe(engine_id, data)
        
        # Preprocess
        processed = self._preprocess_dataframe(df)
        
        # Normalize
        norm_df = self.stats.apply(processed)
        
        # Build inference windows
        windows, unit_ids = build_inference_windows(norm_df, self.sensors, self.seq_len)
        
        # Predict
        tensor = torch.from_numpy(windows).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)
        
        # Return prediction for the engine
        outputs = preds.detach().cpu().numpy()
        return float(outputs[0])

    def _api_data_to_dataframe(self, engine_id: str, data: List[CmapssDataPoint]) -> pd.DataFrame:
        """Convert API data format to DataFrame."""
        # Convert Pydantic models to dictionaries
        rows = [point.model_dump() for point in data]
        df = pd.DataFrame(rows)
        
        # Ensure all required columns exist
        for col in _COL_NAMES:
            if col not in df.columns:
                if col == "unit":
                    # Extract numeric part from engine_id or use 1
                    try:
                        df["unit"] = int(engine_id.split("-")[-1])
                    except (ValueError, IndexError):
                        df["unit"] = 1
                elif col == "cycle":
                    # Use row index as cycle if not provided
                    df["cycle"] = range(len(df))
                else:
                    # Missing sensor or setting - fill with NaN
                    df[col] = np.nan
        
        return df[_COL_NAMES]

    def _preprocess_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame before prediction."""
        missing = [col for col in self._required_columns if col not in raw_df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        
        df = raw_df[self._required_columns].copy()
        
        # Convert to numeric
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

        # Process each unit
        processed_units: List[pd.DataFrame] = []
        for unit_id, unit_df in df.groupby("unit"):
            unit_processed = self._process_unit(unit_id, unit_df)
            processed_units.append(unit_processed)
        processed = pd.concat(processed_units, ignore_index=True)

        # Handle missing values in sensors
        sensor_values = processed[self.sensors].to_numpy(dtype=np.float32)
        sensor_values = self.stats.fill_missing(sensor_values)
        sensor_values = self.stats.clip_outliers(sensor_values)
        sensor_values = self.stats.clip_to_range(sensor_values)
        processed.loc[:, self.sensors] = sensor_values

        # Apply smoothing if configured
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
        """Process individual unit data."""
        unit_df = unit_df.set_index("cycle")
        full_index = np.arange(unit_df.index.min(), unit_df.index.max() + 1)
        unit_df = unit_df.reindex(full_index)
        unit_df["unit"] = unit_id
        
        # Interpolate settings
        unit_df[self._settings] = unit_df[self._settings].interpolate(method="linear", limit_direction="both")
        unit_df[self._settings] = unit_df[self._settings].fillna(0.0)
        
        # Interpolate sensors
        unit_df[self.sensors] = unit_df[self.sensors].interpolate(method="linear", limit_direction="both")
        unit_df[self.sensors] = unit_df[self.sensors].fillna(unit_df[self.sensors].median())
        
        unit_df.reset_index(inplace=True)
        unit_df.rename(columns={"index": "cycle"}, inplace=True)
        return unit_df
