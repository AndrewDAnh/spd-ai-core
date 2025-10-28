"""
One-time script to preprocess C-MAPSS data.
Generates reference and current datasets with realistic issues.

Usage:
    python scripts/preprocess_cmapss.py
    
Output:
    examples/cmapss/reference_baseline.json
    examples/cmapss/current_data.json
    examples/cmapss/reference_stats.json
    examples/cmapss/current_stats.json
"""

import sys
import pathlib
import json

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from app.utils.cmapss_loader import load_test_dataset, get_sensor_columns
from scripts.realistic_transform import (
    add_noise, inject_missing_values, inject_outliers,
    inject_drift, create_anomalous_engines, TRANSFORM_CONFIG
)
from scripts.format_converter import convert_to_api_format, get_data_statistics


def main():
    print("=" * 60)
    print("C-MAPSS Data Preprocessing")
    print("=" * 60)
    
    # Load test data
    print("\n[1/6] Loading test dataset...")
    df = load_test_dataset()
    sensor_cols = get_sensor_columns()
    
    print(f"  ✓ Loaded {len(df)} cycles from {len(df['unit'].unique())} engines")
    print(f"  ✓ Using {len(sensor_cols)} sensors: {sensor_cols[0]} to {sensor_cols[-1]}")
    
    # Split engines: first 50 = reference, last 50 = current
    print("\n[2/6] Splitting engines...")
    reference_engines = list(range(1, 51))
    current_engines = list(range(51, 101))
    
    print(f"  ✓ Reference: Engines 1-50 ({len(reference_engines)} engines)")
    print(f"  ✓ Current: Engines 51-100 ({len(current_engines)} engines)")
    
    # Create reference (no transformation - keep clean)
    print("\n[3/6] Creating reference baseline (no transformation)...")
    ref_df = df[df['unit'].isin(reference_engines)].copy()
    print(f"  Original: {len(ref_df)} cycles")
    print(f"  ✓ Kept clean (no transformations)")
    
    # Create current (realistic transformation)
    print("\n[4/6] Creating current dataset (realistic transformation)...")
    cur_df = df[df['unit'].isin(current_engines)].copy()
    print(f"  Original: {len(cur_df)} cycles")
    
    # Apply transformations with reduced rates
    cur_df = inject_drift(cur_df, ratio=TRANSFORM_CONFIG['drift_ratio'], random_seed=TRANSFORM_CONFIG['random_seed'])
    cur_df = inject_missing_values(cur_df, rate=TRANSFORM_CONFIG['missing_rate'], random_seed=TRANSFORM_CONFIG['random_seed'])
    cur_df = inject_outliers(cur_df, rate=TRANSFORM_CONFIG['outlier_rate'], random_seed=TRANSFORM_CONFIG['random_seed'])
    cur_df = create_anomalous_engines(cur_df, ratio=TRANSFORM_CONFIG['anomalous_ratio'], random_seed=TRANSFORM_CONFIG['random_seed'])
    # Note: No additional noise added (keeping NASA data quality)
    
    print(f"  ✓ Applied realistic transformations")
    
    # Get statistics
    print("\n[5/6] Calculating statistics...")
    ref_stats = get_data_statistics(ref_df, sensor_cols)
    cur_stats = get_data_statistics(cur_df, sensor_cols)
    
    print(f"  Reference: {ref_stats['n_engines']} engines, {ref_stats['total_cycles']} cycles")
    print(f"  Current: {cur_stats['n_engines']} engines, {cur_stats['total_cycles']} cycles")
    
    # Convert to API format
    print("\n[6/6] Converting to API format and saving...")
    ref_json = convert_to_api_format(ref_df, sensor_cols)
    cur_json = convert_to_api_format(cur_df, sensor_cols)
    
    # Save files
    output_dir = pathlib.Path("examples/cmapss")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reference
    ref_path = output_dir / "reference_baseline.json"
    with open(ref_path, 'w') as f:
        json.dump(ref_json, f, indent=2)
    print(f"  ✓ Saved: {ref_path}")
    
    # Save current
    cur_path = output_dir / "current_data.json"
    with open(cur_path, 'w') as f:
        json.dump(cur_json, f, indent=2)
    print(f"  ✓ Saved: {cur_path}")
    
    # Save statistics
    ref_stats_path = output_dir / "reference_stats.json"
    with open(ref_stats_path, 'w') as f:
        json.dump(ref_stats, f, indent=2)
    print(f"  ✓ Saved: {ref_stats_path}")
    
    cur_stats_path = output_dir / "current_stats.json"
    with open(cur_stats_path, 'w') as f:
        json.dump(cur_stats, f, indent=2)
    print(f"  ✓ Saved: {cur_stats_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {ref_path}")
    print(f"  - {cur_path}")
    print(f"  - {ref_stats_path}")
    print(f"  - {cur_stats_path}")
    
    print(f"\nData Quality Summary (Current Data):")
    print(f"  - Engines with drift: ~{int(TRANSFORM_CONFIG['drift_ratio'] * 50)} ({int(TRANSFORM_CONFIG['drift_ratio']*100)}%)")
    print(f"  - Anomalous engines: ~{int(TRANSFORM_CONFIG['anomalous_ratio'] * 50)} ({int(TRANSFORM_CONFIG['anomalous_ratio']*100)}%)")
    print(f"  - Missing values: ~{int(TRANSFORM_CONFIG['missing_rate']*100)}% of data points")
    print(f"  - Outliers: ~{int(TRANSFORM_CONFIG['outlier_rate']*100)}% of data points")
    print(f"  - Additional noise: None (keeping NASA data quality)")
    
    print(f"\nNext steps:")
    print(f"  1. Start the validation API: uvicorn app.main:app --reload")
    print(f"  2. Run validation tests: python scripts/test_cmapss_validation.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

