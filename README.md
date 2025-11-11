# API Documentation

## Overview

The SPD-MVP API helps you keep tabs on jet-engine health through three major capabilities:
- **Inference** – Remaining Useful Life (RUL) predictions from the STAR regression model plus a BiLSTM failure classifier.
- **Validation** – Drift detection and quality checks for incoming sensor feeds.
- **Monitoring** – Model performance baselines, prediction history, and drift alerts.

## Base URL

```
http://localhost:8000/api/v1
```

## Inference Endpoints

### POST /inference/predict/batch

Submit one or more engines and receive both RUL estimates and short-term failure warnings in a single call.

**Request Body:**

```json
{
  "batch_id": "batch_001",
  "engines": [
    {
      "engine_id": "ENG-001",
      "timestamp": "2025-10-26T12:00:00Z",
      "data": [
        {
          "unit": 1,
          "cycle": 1,
          "setting_1": 0.0023,
          "setting_2": 0.0003,
          "setting_3": 100.0,
          "s1": 518.67,
          "s2": 641.82,
          "s3": 1589.70,
          "s4": 1400.60,
          "s5": 14.62,
          "s6": 21.61,
          "s7": 554.36,
          "s8": 2388.06,
          "s9": 9046.19,
          "s10": 1.30,
          "s11": 47.47,
          "s12": 521.66,
          "s13": 2388.02,
          "s14": 8138.62,
          "s15": 8.41,
          "s16": 0.03,
          "s17": 392.00,
          "s18": 2388.00,
          "s19": 100.00,
          "s20": 39.06,
          "s21": 23.42
        }
      ]
    }
  ]
}
```

**Field Descriptions:**

- `batch_id`: Unique identifier for the batch
- `engines`: Array of engine data objects
  - `engine_id`: Unique identifier for the engine
  - `timestamp`: ISO 8601 timestamp of the prediction request
  - `data`: Array of time-step measurements in C-MAPSS format
    - `unit`: Engine unit ID (integer)
    - `cycle`: Cycle number (integer)
    - `setting_1`, `setting_2`, `setting_3`: Operational settings
    - `s1` to `s21`: Sensor readings (all optional, `null` for missing values)

**Important Notes:**

1. **C-MAPSS Format** – Stick to the NASA turbofan schema (3 settings + 21 sensors). We keep optional sensors nullable so validation can detect gaps.
2. **Sequence Length** – 128 time steps give the STAR model the temporal context it expects. Shorter sequences work but you will lose accuracy.
3. **Sensor Subset** – The regression model leans on 14 sensors: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21. Still supply all readings you have; we handle the rest internally.
4. **Classification Threshold** – The BiLSTM marks `is_going_to_fail` as `true` when failure probability ≥ 0.5. Future releases will let you tune this threshold per request.

**Response:**

```json
{
  "prediction_id": "pred_abc123def456",
  "batch_id": "batch_001",
  "timestamp": "2025-10-26T12:00:01.234567",
  "predictions": [
    {
      "engine_id": "ENG-001",
      "prediction_time": "2025-10-26T12:00:01.234567",
      "remaining_useful_life": 85.34,
      "is_going_to_fail": false,
      "confidence": 0.87
    }
  ]
}
```

**Response Field Descriptions:**

- `remaining_useful_life`: Predicted RUL in cycles (float)
- `is_going_to_fail`: `true` when the BiLSTM classifier believes failure is imminent based on a 0.5 probability cut-off. This field becomes `null` only if the classifier fails to load at startup.
- `confidence`: Regression-side confidence heuristic ranging from 0.0 to 1.0. High values mean the RUL falls squarely in healthy or critical territory; mid-range values are more ambiguous.

**Status Codes:**

- `200 OK`: Prediction successful
- `422 Unprocessable Entity`: Invalid request format
- `500 Internal Server Error`: Model inference error

### GET /inference/predict/history/{engine_id}

Retrieve the stored predictions (regression + classification + confidence) for a specific engine.

**Query Parameters:**

- `start_date` (optional): Filter predictions after this date (ISO 8601)
- `end_date` (optional): Filter predictions before this date (ISO 8601)
- `limit` (optional): Maximum number of predictions to return (default: 100)

**Example:**

```
GET /api/v1/inference/predict/history/ENG-001?limit=50
```

**Response:**

```json
{
  "engine_id": "ENG-001",
  "predictions": [
    {
      "engine_id": "ENG-001",
      "prediction_time": "2025-10-26T12:00:00Z",
      "remaining_useful_life": 85.34,
      "is_going_to_fail": false,
      "confidence": 0.87
    }
  ],
  "total_count": 50
}
```

## Validation Endpoints

- **POST /validate/batch** – Run drift and quality checks together. Optional overrides allow to tweak PSI thresholds or skip stored baselines.
- **POST /validate/drift** – Compare incoming feature distributions against the reference baseline using PSI and KS tests.
- **POST /validate/quality** – Flag missing values, constant columns, and statistical anomalies without running drift.
- **POST /validate/reference** – Upload or refresh a reference baseline. Useful when an engine changes operating regimes.
- **GET /validate/summary** – Quick snapshot of recent validations (counts of high/medium issues, drift detections, engines monitored).

## Performance & Monitoring Endpoints

- **POST /models/performance/run** – Executes the full FD001 benchmark: STAR regression metrics (MSE, MAE, MAPE) plus classification precision/recall at six probability thresholds (0.0 → 1.0 in steps of 0.2). Results are stored for later comparison.
- **GET /models/performance** – Fetch the most recent performance record, including MAE, MSE, MAPE for regression model, and precision, recall, f1 for classification model.
- **POST /validate/model-drift** – Evaluate prediction stability over time per engine for RUL prediction. Formula is calculated as `|RUL_current - RUL_previous| / hours_elapsed`
- **GET /inference/predict/history/{engine_id}** – Already covered above; handy for ad-hoc drift investigations.

## Health Check

### GET /health

Check API health status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-10-26T12:00:00.123456",
  "version": "1.0.0"
}
```

## Model Information

### STAR Regression Model

- **Architecture** – Multi-scale transformer with temporal and sensor attention stages.
- **Input Window** – 128 × 14 feature matrix after sensor selection and normalization.
- **Output** – Continuous RUL prediction in cycles.
- **Training Data** – C-MAPSS FD001 split (same format as inference requests).

### BiLSTM Classification Model

- **Goal** – Binary early-warning signal (`is_going_to_fail`) telling whether the engine is going to fail within the next 30 cycles.
- **Output Surface** – Currently exposes the boolean decision via the API. The raw probability feeds multi-threshold metrics in the performance service.

## Error Handling

All endpoints return standard HTTP status codes and error messages:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Starting Application

Use the provided script:

```bash
# Start the API
uvicorn app.main:app --host 0.0.0.0 --port [YOUR_PORT_NUMBER]
```

## Examples

### Complete Example: Single Engine Prediction

```python
import requests
from datetime import datetime

# Prepare request
url = f"http://{HOST}:{PORT}/api/v1/inference/predict/batch"
data = {
    "batch_id": "test_001",
    "engines": [{
        "engine_id": "ENG-001",
        "timestamp": datetime.now(UTC.isoformat() + "Z",
    "data": [
      {
        "unit": 1,
        "cycle": i,
        "setting_1": 0.0023,
        "setting_2": 0.0003,
        "setting_3": 100.0,
        "s1": 642.0,
        "s2": 1580.0
        # Populate remaining sensors as needed
      }
      for i in range(128)
    ]
    }]
}

# Make request
response = requests.post(url, json=data)
result = response.json()

# Extract prediction
pred = result['predictions'][0]
print(f"Predicted RUL: {pred['remaining_useful_life']:.2f} cycles")
print(f"Failure warning: {pred['is_going_to_fail']}")
```

## Integration with Validation API

The inference API can be used together with the validation API:

1. **Predict** – Call `/inference/predict/batch` to score a fleet.
2. **Validate** – Run `/validate/batch` on the same payload to spot quality or distribution issues.
3. **Monitor** – Use `/models/performance` for periodic benchmarks and `/validate/summary` for day-to-day oversight.