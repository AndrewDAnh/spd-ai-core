# C-MAPSS Preprocessed Data

This directory contains preprocessed C-MAPSS FD001 test data for validation API testing.

## Files

### Generated Files (after running preprocess_cmapss.py)

- **reference_baseline.json** - Clean reference data (engines 1-50)
  - Minimal transformations
  - Used as baseline for drift detection
  - ~50 engines, ~5000-7000 total data points

- **current_data.json** - Transformed current data (engines 51-100)
  - Full transformations applied
  - Contains realistic data quality issues
  - ~50 engines, ~5000-7000 total data points

- **reference_stats.json** - Statistics for reference data
  - Missing value percentages
  - Mean, std, min, max per sensor
  - Overall data quality metrics

- **current_stats.json** - Statistics for current data
  - Shows impact of transformations
  - Comparison with reference baseline

## How to Generate

```bash
# From project root
python3 scripts/preprocess_cmapss.py
```

## Data Transformations Applied (Current Data Only)

1. **Noise**: 2x multiplier (more realistic)
2. **Drift**: 25% of engines have calibration drift
3. **Missing Values**: 10% rate, higher in flow sensors
4. **Outliers**: 3% rate in temp/pressure sensors
5. **Anomalous Engines**: 8% engines with multiple issues

## Data Format

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

## Usage with Validation API

### Store Reference

```python
import requests

with open("reference_baseline.json") as f:
    reference = json.load(f)

# Store first engine as reference
requests.post(
    "http://localhost:8000/api/v1/validate/reference",
    json={
        "engine_id": reference["engines"][0]["engine_id"],
        "reference_data": reference["engines"][0]["data"]
    }
)
```

### Validate Current Data

```python
with open("current_data.json") as f:
    current = json.load(f)

# Validate against stored reference
response = requests.post(
    "http://localhost:8000/api/v1/validate/batch",
    json={
        "validation_id": "cmapss_test",
        "engines": current["engines"][:10],
        "use_stored_reference": True,
        "config": {
            "drift_threshold": 0.2,
            "outlier_sensitivity": "medium"
        }
    }
)
```

## Expected Results

- **Drift Detection**: ~12-15 engines with drift (out of 50)
- **Quality Issues**: ~18-22 engines with quality problems
- **High Severity**: ~5-8 engines
- **Medium Severity**: ~10-15 engines

## Notes

- Data is generated from NASA C-MAPSS FD001 test set
- Transformations are reproducible (random_seed=42)
- Files are ~10-50MB each (depending on compression)
- Safe to delete and regenerate anytime

## Regenerate Data

To regenerate with different parameters:

1. Edit `scripts/realistic_transform.py`
2. Modify `TRANSFORM_CONFIG` dictionary
3. Run `python3 scripts/preprocess_cmapss.py`

---

For more details, see `docs/CMAPSS_INTEGRATION.md`

