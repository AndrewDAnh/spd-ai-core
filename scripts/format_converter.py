"""Convert C-MAPSS DataFrame to validation API JSON format"""

import pandas as pd
import numpy as np
from typing import List, Dict
import sys
import pathlib

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def convert_to_api_format(df: pd.DataFrame, sensor_cols: List[str]) -> dict:
    """Convert C-MAPSS data to validation API format
    
    Maps C-MAPSS sensor names (s1, s2, ...) to API format (sensor_1, sensor_2, ...)
    and structures data for validation endpoints.
    
    Args:
        df: DataFrame with C-MAPSS data (unit, cycle, s1-s21, etc.)
        sensor_cols: List of sensor column names to include (e.g., ['s1', 's2', ...])
        
    Returns:
        Dictionary in validation API format:
        {
            "engines": [
                {
                    "engine_id": "ENG-001",
                    "data": [
                        {"sensor_1": 518.67, "sensor_2": 641.82, ...},
                        ...
                    ]
                },
                ...
            ]
        }
    
    Note:
        Missing values (NaN) are preserved as null in JSON to allow 
        quality checks to detect them.
    """
    engines = []
    
    for engine_id in sorted(df['unit'].unique()):
        engine_df = df[df['unit'] == engine_id].copy()
        
        # Convert each cycle to a data point
        data_points = []
        for _, row in engine_df.iterrows():
            point = {}
            for i, col in enumerate(sensor_cols, 1):
                # Map s1 -> sensor_1, s2 -> sensor_2, etc.
                value = row[col]
                # Preserve NaN as None (null in JSON) for quality checks
                if pd.isna(value):
                    point[f"sensor_{i}"] = None
                else:
                    point[f"sensor_{i}"] = float(value)
            data_points.append(point)
        
        engines.append({
            "engine_id": f"ENG-{engine_id:03d}",  # ENG-001, ENG-002, etc.
            "data": data_points
        })
    
    return {"engines": engines}


def create_validation_request(engines_data: dict, validation_id: str,
                              use_stored_reference: bool = True,
                              drift_threshold: float = 0.2,
                              outlier_sensitivity: str = "medium") -> dict:
    """Create a validation request payload
    
    Args:
        engines_data: Output from convert_to_api_format
        validation_id: Unique ID for this validation
        use_stored_reference: Whether to use stored reference baseline
        drift_threshold: Threshold for drift detection
        outlier_sensitivity: Sensitivity for outlier detection (low/medium/high)
        
    Returns:
        Validation request dictionary
    """
    return {
        "validation_id": validation_id,
        "engines": engines_data["engines"],
        "use_stored_reference": use_stored_reference,
        "config": {
            "drift_threshold": drift_threshold,
            "outlier_sensitivity": outlier_sensitivity
        }
    }


def create_reference_request(engine_data: dict) -> dict:
    """Create a reference baseline storage request
    
    Args:
        engine_data: Single engine data from convert_to_api_format["engines"][i]
        
    Returns:
        Reference request dictionary
    """
    return {
        "engine_id": engine_data["engine_id"],
        "reference_data": engine_data["data"]
    }


def get_data_statistics(df: pd.DataFrame, sensor_cols: List[str]) -> dict:
    """Get statistics about the transformed data
    
    Args:
        df: DataFrame with data
        sensor_cols: List of sensor columns
        
    Returns:
        Dictionary with statistics (JSON-serializable)
    """
    stats = {
        "n_engines": int(len(df['unit'].unique())),
        "total_cycles": int(len(df)),
        "avg_cycles_per_engine": float(len(df) / len(df['unit'].unique())),
        "sensor_stats": {}
    }
    
    for sensor in sensor_cols:
        if sensor not in df.columns:
            continue
            
        sensor_data = df[sensor]
        missing_pct = (sensor_data.isna().sum() / len(sensor_data)) * 100
        
        stats["sensor_stats"][sensor] = {
            "missing_pct": float(round(missing_pct, 2)),
            "mean": float(round(sensor_data.mean(), 2)) if not sensor_data.isna().all() else None,
            "std": float(round(sensor_data.std(), 2)) if not sensor_data.isna().all() else None,
            "min": float(round(sensor_data.min(), 2)) if not sensor_data.isna().all() else None,
            "max": float(round(sensor_data.max(), 2)) if not sensor_data.isna().all() else None
        }
    
    return stats

