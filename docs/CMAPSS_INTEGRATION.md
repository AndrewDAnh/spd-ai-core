# C-MAPSS Dataset Integration Guide

## Overview

This guide explains how to use the NASA C-MAPSS FD001 dataset with the validation API for testing drift detection, quality checks, and model drift monitoring.

## Dataset Information

**C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
- Source: NASA Ames Prognostics Data Repository
- Dataset: FD001 (Fault Dataset 001)
- Test engines: 100
- Sensors: 21 sensors + 3 operational settings
- Format: Space-separated text file

**Files:**
- `test_FD001.txt`: 100 test engine run-to-failure trajectories
- `RUL_FD001.txt`: Ground truth RUL values for each test engine
- `readme.txt`: Dataset description

## Architecture

```
test_FD001.txt (clean NASA data)
         ↓
[One-Time Preprocessing Scripts]
  - Load data
  - Apply realistic transformations
  - Split into reference/current
  - Convert to API format
         ↓
reference_baseline.json + current_data.json
         ↓
[Validation API]
  - Drift detection
  - Quality checks
  - Model drift tracking
         ↓
Validation Results
```

## Quick Start

### Step 1: Install Dependencies

```bash
cd spd-mvp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Preprocess Data

```bash
python3 scripts/preprocess_cmapss.py
```

**Output:**
- `examples/cmapss/reference_baseline.json` - Clean reference data (engines 1-50)
- `examples/cmapss/current_data.json` - Transformed data with issues (engines 51-100)
- `examples/cmapss/reference_stats.json` - Reference statistics
- `examples/cmapss/current_stats.json` - Current statistics

### Step 3: Start API

```bash
uvicorn app.main:app --reload
```

### Step 4: Run Tests

```bash
python3 scripts/test_cmapss_validation.py
```

## Data Transformations

The preprocessing applies realistic issues to simulate production data:

### 1. **Missing Values** (10% rate)
- Simulates sensor failures
- Higher rates in flow sensors (s11, s12, s13)
- Increases in later cycles (sensor degradation)

### 2. **Outliers** (3% rate)
- Simulates sensor malfunctions
- Types: spike outliers, stuck values
- Targets temperature and pressure sensors

### 3. **Drift** (25% of engines)
- Simulates calibration drift
- Linear drift patterns
- Temperature/pressure sensors most affected

### 4. **Noise** (2x multiplier)
- NASA data too clean for realistic testing
- Adds Gaussian noise to all sensors

### 5. **Anomalous Engines** (8% of engines)
- Highly problematic engines
- Multiple issues simultaneously
- 20% missing + 5% outliers

## File Structure

```
spd-mvp/
├── datasets/
│   ├── test_FD001.txt           # NASA test data
│   ├── RUL_FD001.txt            # Ground truth
│   └── readme.txt
├── app/utils/
│   ├── dataset.py               # Original utilities
│   └── cmapss_loader.py         # NEW: C-MAPSS loader
├── scripts/                     # NEW: Preprocessing
│   ├── realistic_transform.py   # Transformation functions
│   ├── format_converter.py      # Format conversion
│   ├── preprocess_cmapss.py     # Main preprocessing
│   └── test_cmapss_validation.py # Testing script
└── examples/cmapss/             # NEW: Preprocessed data
    ├── reference_baseline.json
    ├── current_data.json
    ├── reference_stats.json
    └── current_stats.json
```

## Components

### 1. cmapss_loader.py

**Purpose:** Load C-MAPSS data using existing dataset.py utilities

**Functions:**
- `load_test_dataset()` - Load test_FD001.txt
- `load_rul_values()` - Load RUL_FD001.txt
- `extract_engine_data()` - Extract specific engine
- `get_sensor_columns()` - Get sensor column names

### 2. realistic_transform.py

**Purpose:** Transform clean data into realistic production-like data

**Functions:**
- `inject_missing_values()` - Add missing values
- `inject_outliers()` - Add outliers (spikes, stuck values)
- `inject_drift()` - Add calibration drift
- `add_noise()` - Increase noise levels
- `create_anomalous_engines()` - Create problematic engines

**Configuration:**
```python
TRANSFORM_CONFIG = {
    'missing_rate': 0.10,      # 10% missing
    'outlier_rate': 0.03,      # 3% outliers
    'drift_ratio': 0.25,       # 25% engines with drift
    'noise_multiplier': 2.0,   # 2x noise
    'anomalous_ratio': 0.08,   # 8% anomalous engines
    'random_seed': 42          # Reproducibility
}
```

### 3. format_converter.py

**Purpose:** Convert DataFrame to API JSON format

**Functions:**
- `convert_to_api_format()` - Main conversion
- `create_validation_request()` - Create request payload
- `create_reference_request()` - Create reference payload
- `get_data_statistics()` - Calculate statistics

**Format Mapping:**
- `s1` → `sensor_1`
- `s2` → `sensor_2`
- ... → ...
- `s21` → `sensor_21`

### 4. preprocess_cmapss.py

**Purpose:** Main preprocessing pipeline

**Process:**
1. Load test_FD001.txt
2. Split: engines 1-50 (reference), 51-100 (current)
3. Reference: minimal noise only
4. Current: all transformations
5. Convert to API format
6. Save JSON files + statistics

### 5. test_cmapss_validation.py

**Purpose:** End-to-end validation testing

**Tests:**
1. Health check
2. Store reference baselines
3. Batch validation (drift + quality)
4. Drift detection only
5. Quality check only
6. Validation summary

## Data Format

### Input (C-MAPSS)

```
unit cycle setting_1 setting_2 setting_3 s1 s2 ... s21
1    1     -0.0007   -0.0004   100.0     518.67 641.82 ... 23.4190
1    2     0.0019    -0.0003   100.0     518.67 642.15 ... 23.4236
...
```

### Output (API Format)

```json
{
  "engines": [
    {
      "engine_id": "ENG-001",
      "data": [
        {
          "sensor_1": 518.67,
          "sensor_2": 641.82,
          ...
          "sensor_21": 23.4190
        },
        ...
      ]
    },
    ...
  ]
}
```

## Usage Examples

### Load Data

```python
from app.utils.cmapss_loader import load_test_dataset, get_sensor_columns

# Load test dataset
df = load_test_dataset()
sensor_cols = get_sensor_columns()

print(f"Loaded {len(df)} cycles from {len(df['unit'].unique())} engines")
```

### Apply Transformations

```python
from scripts.realistic_transform import inject_missing_values, inject_drift

# Apply transformations
df_transformed = inject_missing_values(df, rate=0.10)
df_transformed = inject_drift(df_transformed, ratio=0.25)
```

### Convert to API Format

```python
from scripts.format_converter import convert_to_api_format

# Convert to API format
api_data = convert_to_api_format(df, sensor_cols)
print(f"Converted {len(api_data['engines'])} engines")
```

### Test Validation

```python
import requests

# Store reference
response = requests.post(
    "http://localhost:8000/api/v1/validate/reference",
    json={
        "engine_id": "ENG-001",
        "reference_data": reference_data
    }
)

# Validate current data
response = requests.post(
    "http://localhost:8000/api/v1/validate/batch",
    json={
        "validation_id": "test_001",
        "engines": current_data,
        "use_stored_reference": True
    }
)
```

## Expected Results

After preprocessing and validation:

### Drift Detection

**Expected:**
- Drift detected in ~25% of current engines
- Higher drift scores in temperature/pressure sensors
- Overall drift score: 0.15-0.30

**Example Output:**
```json
{
  "drift_detected": true,
  "overall_drift_score": 0.22,
  "engines_with_drift": ["ENG-063", "ENG-072", "ENG-085", ...]
}
```

### Quality Checks

**Expected:**
- ~18 engines with issues (out of 50)
- ~10% missing values detected
- ~3% outliers detected
- 5-8 high severity issues

**Example Output:**
```json
{
  "quality_passed": false,
  "total_issues": 45,
  "engines_with_issues": 18,
  "high_severity_count": 5
}
```

## Validation Endpoints

All validation endpoints work without modification:

### Store Reference
```bash
POST /api/v1/validate/reference
```

### Batch Validation
```bash
POST /api/v1/validate/batch
```

### Drift Detection
```bash
POST /api/v1/validate/drift
```

### Quality Check
```bash
POST /api/v1/validate/quality
```

### Validation Summary
```bash
GET /api/v1/validate/summary
```

## Troubleshooting

### Module Not Found

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### API Not Running

```bash
# Start the API
uvicorn app.main:app --reload

# Check health
curl http://localhost:8000/api/v1/health
```

### Data Not Found

```bash
# Run preprocessing first
python3 scripts/preprocess_cmapss.py

# Check output
ls examples/cmapss/
```

## Technical Notes

### Sensor Selection

C-MAPSS has 21 sensors. Key sensors:

- **s2, s3, s4:** Temperature sensors (prone to drift)
- **s7, s8, s9:** Pressure sensors (prone to outliers)
- **s11, s12, s13:** Flow sensors (prone to missing values)
- **s20, s21:** Vibration sensors (high noise)

### Cycle Numbers

- Cycles represent operational hours or flight cycles
- No datetime mapping (kept as integers)
- Variable length per engine (100-200 cycles in test set)

### Sequence Length

- Transformer model uses seq_len=128
- Preprocessing works with full sequences
- For inference: use `build_inference_windows()` from dataset.py

### Random Seed

- All transformations use seed=42 for reproducibility
- Same preprocessing always produces same output
- Change seed in TRANSFORM_CONFIG for different patterns

## Integration with Inference Pipeline

**Future:** When adding inference capabilities:

```python
from app.utils.dataset import build_inference_windows

# Window sequences for model input
windows, unit_ids = build_inference_windows(
    df, 
    sensors=get_sensor_columns(),
    seq_len=128
)
```

## References

- NASA C-MAPSS Dataset: [NASA Prognostics Repository]
- Paper: Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08

## Next Steps

1. ✅ Preprocess data: `python3 scripts/preprocess_cmapss.py`
2. ✅ Start API: `uvicorn app.main:app --reload`
3. ✅ Run tests: `python3 scripts/test_cmapss_validation.py`
4. ✅ Review results and adjust thresholds
5. ✅ Integrate into production workflow

---

**Questions?** Check the main README.md or test_api.py for more examples.

