"""C-MAPSS data loading utilities (reuses dataset.py logic)"""

from app.utils.dataset import _read_raw, _COL_NAMES
import pandas as pd
import pathlib
from typing import Dict, List


def load_test_dataset(dataset_dir: str = "datasets") -> pd.DataFrame:
    """Load test_FD001.txt using dataset.py utilities
    
    Args:
        dataset_dir: Directory containing the dataset files
        
    Returns:
        DataFrame with columns: unit, cycle, setting_1, setting_2, setting_3, s1-s21
    """
    path = pathlib.Path(dataset_dir) / "test_FD001.txt"
    return _read_raw(path)


def load_rul_values(dataset_dir: str = "datasets") -> pd.DataFrame:
    """Load RUL_FD001.txt with ground truth RUL values
    
    Args:
        dataset_dir: Directory containing the dataset files
        
    Returns:
        DataFrame with column: rul (one value per engine)
    """
    path = pathlib.Path(dataset_dir) / "RUL_FD001.txt"
    return pd.read_csv(path, header=None, names=["rul"])


def extract_engine_data(df: pd.DataFrame, engine_id: int, 
                        sensor_cols: List[str]) -> pd.DataFrame:
    """Extract specific engine's sensor data
    
    Args:
        df: Full dataset DataFrame
        engine_id: Unit number to extract
        sensor_cols: List of sensor column names to include
        
    Returns:
        DataFrame with only the specified engine and sensor columns
    """
    return df[df["unit"] == engine_id][sensor_cols]


def get_sensor_columns() -> List[str]:
    """Return sensor column names (s1 to s21)
    
    Returns:
        List of sensor column names
    """
    return [f"s{i}" for i in range(1, 22)]


def get_all_engine_ids(df: pd.DataFrame) -> List[int]:
    """Get list of all engine IDs in the dataset
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        Sorted list of engine IDs
    """
    return sorted(df["unit"].unique().tolist())


def get_engine_cycle_count(df: pd.DataFrame, engine_id: int) -> int:
    """Get number of cycles for a specific engine
    
    Args:
        df: Dataset DataFrame
        engine_id: Unit number
        
    Returns:
        Number of cycles for the engine
    """
    return len(df[df["unit"] == engine_id])
