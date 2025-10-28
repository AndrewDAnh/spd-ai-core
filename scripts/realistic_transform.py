"""
Data quality transformation functions.
These are used ONCE to generate realistic test data.
NOT part of the production service.
"""

import pandas as pd
import numpy as np
from typing import List, Set
import sys
import pathlib

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app.utils.cmapss_loader import get_sensor_columns


# Configuration
TRANSFORM_CONFIG = {
    'missing_rate': 0.03,      # 3% missing
    'outlier_rate': 0.03,      # 3% outliers
    'drift_ratio': 0.12,       # 12% engines with drift
    'noise_multiplier': 0.0,   # No additional noise
    'anomalous_ratio': 0.04,   # 4% anomalous engines
    'random_seed': 42
}


def inject_missing_values(df: pd.DataFrame, rate: float = 0.10, 
                         target_sensors: List[str] = None,
                         random_seed: int = 42) -> pd.DataFrame:
    """Inject missing values (one-time preprocessing)
    
    Simulates sensor failures by randomly removing values.
    More aggressive in later cycles to simulate sensor degradation.
    
    Args:
        df: DataFrame with sensor data
        rate: Base missing value rate (0-1)
        target_sensors: Sensors to target (defaults to s11, s12, s13 - flow sensors)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with missing values injected
    """
    np.random.seed(random_seed)
    df_copy = df.copy()
    
    if target_sensors is None:
        # Target flow sensors which are more prone to failures
        sensor_cols = get_sensor_columns()
        target_sensors = ['s11', 's12', 's13', 's14', 's15']
        target_sensors = [s for s in target_sensors if s in sensor_cols]
    
    for engine_id in df_copy['unit'].unique():
        engine_mask = df_copy['unit'] == engine_id
        engine_data = df_copy[engine_mask]
        n_cycles = len(engine_data)
        
        for sensor in target_sensors:
            if sensor not in df_copy.columns:
                continue
                
            # Increase missing rate in later cycles (degradation)
            for idx, (orig_idx, row) in enumerate(engine_data.iterrows()):
                degradation_factor = 1 + (idx / n_cycles) * 0.5  # Up to 1.5x more missing at end
                adjusted_rate = min(rate * degradation_factor, 0.3)  # Cap at 30%
                
                if np.random.random() < adjusted_rate:
                    df_copy.at[orig_idx, sensor] = np.nan
    
    return df_copy


def inject_outliers(df: pd.DataFrame, rate: float = 0.03,
                   target_sensors: List[str] = None,
                   random_seed: int = 42) -> pd.DataFrame:
    """Inject outliers (one-time preprocessing)
    
    Simulates sensor malfunctions with spike outliers and stuck values.
    
    Args:
        df: DataFrame with sensor data
        rate: Outlier rate (0-1)
        target_sensors: Sensors to target (defaults to temp/pressure sensors)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with outliers injected
    """
    np.random.seed(random_seed + 1)  # Different seed
    df_copy = df.copy()
    
    if target_sensors is None:
        # Target temperature and pressure sensors
        sensor_cols = get_sensor_columns()
        target_sensors = ['s2', 's3', 's4', 's7', 's8', 's9']
        target_sensors = [s for s in target_sensors if s in sensor_cols]
    
    for sensor in target_sensors:
        if sensor not in df_copy.columns:
            continue
            
        # Calculate sensor statistics
        sensor_mean = df_copy[sensor].mean()
        sensor_std = df_copy[sensor].std()
        sensor_range = df_copy[sensor].max() - df_copy[sensor].min()
        
        # Determine outlier indices
        n_total = len(df_copy)
        n_outliers = int(n_total * rate)
        outlier_indices = np.random.choice(df_copy.index, size=n_outliers, replace=False)
        
        for idx in outlier_indices:
            outlier_type = np.random.choice(['spike_high', 'spike_low', 'stuck'])
            
            if outlier_type == 'spike_high':
                # Spike 4-6 sigma above mean
                df_copy.at[idx, sensor] = sensor_mean + np.random.uniform(4, 6) * sensor_std
            elif outlier_type == 'spike_low':
                # Spike 4-6 sigma below mean
                df_copy.at[idx, sensor] = sensor_mean - np.random.uniform(4, 6) * sensor_std
            else:  # stuck
                # Stuck at random value (simulate sensor freeze)
                stuck_value = df_copy[sensor].sample(1).values[0]
                # Make consecutive values stuck
                for offset in range(np.random.randint(3, 8)):
                    if idx + offset < len(df_copy):
                        df_copy.at[idx + offset, sensor] = stuck_value
    
    return df_copy


def inject_drift(df: pd.DataFrame, ratio: float = 0.25,
                target_sensors: List[str] = None,
                random_seed: int = 42) -> pd.DataFrame:
    """Inject drift patterns (one-time preprocessing)
    
    Simulates calibration drift by adding gradual linear trend to sensor readings.
    
    Args:
        df: DataFrame with sensor data
        ratio: Ratio of engines to apply drift to (0-1)
        target_sensors: Sensors to target (defaults to temp/pressure sensors)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with drift injected
    """
    np.random.seed(random_seed + 2)  # Different seed
    df_copy = df.copy()
    
    if target_sensors is None:
        # Target temperature and pressure sensors (prone to calibration drift)
        sensor_cols = get_sensor_columns()
        target_sensors = ['s2', 's3', 's4', 's7', 's8']
        target_sensors = [s for s in target_sensors if s in sensor_cols]
    
    # Select engines for drift
    all_engines = df_copy['unit'].unique()
    n_drift_engines = int(len(all_engines) * ratio)
    drift_engines = np.random.choice(all_engines, size=n_drift_engines, replace=False)
    
    for engine_id in drift_engines:
        engine_mask = df_copy['unit'] == engine_id
        engine_data = df_copy[engine_mask]
        n_cycles = len(engine_data)
        
        # Select random sensors for this engine
        n_sensors_to_drift = np.random.randint(1, min(4, len(target_sensors) + 1))
        sensors_to_drift = np.random.choice(target_sensors, size=n_sensors_to_drift, replace=False)
        
        for sensor in sensors_to_drift:
            if sensor not in df_copy.columns:
                continue
                
            # Calculate drift parameters
            sensor_std = df_copy[sensor].std()
            drift_magnitude = np.random.uniform(0.5, 2.0) * sensor_std  # Drift up to 2 std devs
            drift_direction = np.random.choice([-1, 1])  # Drift up or down
            
            # Apply linear drift
            drift_pattern = np.linspace(0, drift_magnitude * drift_direction, n_cycles)
            
            for idx, (orig_idx, _) in enumerate(engine_data.iterrows()):
                df_copy.at[orig_idx, sensor] += drift_pattern[idx]
    
    return df_copy


def add_noise(df: pd.DataFrame, multiplier: float = 2.0,
             random_seed: int = 42) -> pd.DataFrame:
    """Add realistic noise (one-time preprocessing)
    
    NASA data is very clean. This adds realistic sensor noise.
    
    Args:
        df: DataFrame with sensor data
        multiplier: Noise multiplier (1.0 = original level)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with added noise
    """
    np.random.seed(random_seed + 3)  # Different seed
    df_copy = df.copy()
    
    sensor_cols = get_sensor_columns()
    
    for sensor in sensor_cols:
        if sensor not in df_copy.columns:
            continue
            
        # Estimate current noise level (std of differences)
        sensor_diff = df_copy[sensor].diff().dropna()
        current_noise_std = sensor_diff.std()
        
        # Add Gaussian noise
        noise_std = current_noise_std * (multiplier - 1.0)  # Additional noise
        noise = np.random.normal(0, noise_std, size=len(df_copy))
        
        df_copy[sensor] += noise
    
    return df_copy


def create_anomalous_engines(df: pd.DataFrame, ratio: float = 0.08,
                            random_seed: int = 42) -> pd.DataFrame:
    """Mark and transform anomalous engines
    
    Creates highly problematic engines with multiple issues.
    These represent real problem cases in a fleet.
    
    Args:
        df: DataFrame with sensor data
        ratio: Ratio of engines to make anomalous (0-1)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with anomalous engines
    """
    np.random.seed(random_seed + 4)  # Different seed
    df_copy = df.copy()
    
    # Select anomalous engines
    all_engines = df_copy['unit'].unique()
    n_anomalous = int(len(all_engines) * ratio)
    anomalous_engines = np.random.choice(all_engines, size=n_anomalous, replace=False)
    
    for engine_id in anomalous_engines:
        engine_mask = df_copy['unit'] == engine_id
        
        # Apply aggressive transformations
        # High missing rate
        engine_data = df_copy[engine_mask]
        for sensor in get_sensor_columns():
            if sensor in df_copy.columns:
                missing_mask = np.random.random(len(engine_data)) < 0.20  # 20% missing
                df_copy.loc[engine_mask & df_copy.index.isin(engine_data[missing_mask].index), sensor] = np.nan
        
        # Multiple outliers
        sensor_cols = [s for s in get_sensor_columns() if s in df_copy.columns]
        for sensor in np.random.choice(sensor_cols, size=min(5, len(sensor_cols)), replace=False):
            sensor_mean = df_copy.loc[engine_mask, sensor].mean()
            sensor_std = df_copy.loc[engine_mask, sensor].std()
            
            # Add many outliers
            n_outliers = int(len(engine_data) * 0.05)  # 5% outliers
            outlier_indices = engine_data.sample(n=n_outliers).index
            
            for idx in outlier_indices:
                df_copy.at[idx, sensor] = sensor_mean + np.random.choice([-1, 1]) * np.random.uniform(5, 8) * sensor_std
    
    return df_copy


def apply_all_transformations(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Apply all transformations with given configuration
    
    Args:
        df: Clean DataFrame
        config: Configuration dict (uses TRANSFORM_CONFIG if None)
        
    Returns:
        Transformed DataFrame
    """
    if config is None:
        config = TRANSFORM_CONFIG
    
    df_transformed = df.copy()
    
    print("Applying transformations...")
    print(f"  - Adding noise (multiplier: {config['noise_multiplier']})")
    df_transformed = add_noise(df_transformed, 
                               multiplier=config['noise_multiplier'],
                               random_seed=config['random_seed'])
    
    print(f"  - Injecting drift ({int(config['drift_ratio']*100)}% of engines)")
    df_transformed = inject_drift(df_transformed,
                                  ratio=config['drift_ratio'],
                                  random_seed=config['random_seed'])
    
    print(f"  - Injecting missing values ({int(config['missing_rate']*100)}% rate)")
    df_transformed = inject_missing_values(df_transformed,
                                          rate=config['missing_rate'],
                                          random_seed=config['random_seed'])
    
    print(f"  - Injecting outliers ({int(config['outlier_rate']*100)}% rate)")
    df_transformed = inject_outliers(df_transformed,
                                    rate=config['outlier_rate'],
                                    random_seed=config['random_seed'])
    
    print(f"  - Creating anomalous engines ({int(config['anomalous_ratio']*100)}% of engines)")
    df_transformed = create_anomalous_engines(df_transformed,
                                             ratio=config['anomalous_ratio'],
                                             random_seed=config['random_seed'])
    
    return df_transformed

