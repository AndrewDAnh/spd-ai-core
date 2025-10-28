# API Documentation

## Overview

The SPD-MVP API provides endpoints for:
- **Inference**: RUL (Remaining Useful Life) prediction using STAR regression model
- **Validation**: Data drift detection and quality checks
- **Monitoring**: Model drift tracking and prediction history

## Base URL

```
http://localhost:8000/api/v1
```

## Inference Endpoints

### POST /inference/predict/batch

Batch prediction endpoint for multiple engines using the STAR regression model.

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

1. **Data Format**: The API accepts data in C-MAPSS (NASA turbofan dataset) format
2. **Sequence Length**: The STAR model expects 128 time steps for optimal performance
3. **Missing Values**: Sensor readings can be `null` to indicate missing data
4. **Required Sensors**: The model uses 14 specific sensors (s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21)

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
      "is_going_to_fail": null,
      "confidence": 0.87
    }
  ]
}
```

**Response Field Descriptions:**

- `remaining_useful_life`: Predicted RUL in cycles (float)
- `is_going_to_fail`: Boolean indicating failure risk (currently `null` - classifier not yet available)
- `confidence`: Prediction confidence score (0.0 to 1.0)

**Status Codes:**

- `200 OK`: Prediction successful
- `422 Unprocessable Entity`: Invalid request format
- `500 Internal Server Error`: Model inference error

### GET /inference/predict/history/{engine_id}

Get prediction history for a specific engine.

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
      "is_going_to_fail": null,
      "confidence": 0.87
    }
  ],
  "total_count": 50
}
```

## Validation Endpoints

### POST /validate/batch

Comprehensive validation including drift detection and quality checks.

### POST /validate/drift

Drift detection only.

### POST /validate/quality

Data quality checks only.

### POST /validate/reference

Store reference baseline for drift detection.

### GET /validate/summary

Get validation summary statistics.

See [CMAPSS_INTEGRATION.md](./CMAPSS_INTEGRATION.md) for detailed validation API documentation.

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

The inference API uses the STAR (transformer-based) model:

- **Architecture**: Multi-scale transformer with two-stage attention (temporal + sensor)
- **Input**: 128 time steps × 14 sensors
- **Output**: RUL prediction (cycles remaining)
- **Training Dataset**: C-MAPSS FD001 (NASA turbofan degradation)
- **Performance**: State-of-the-art accuracy on turbofan RUL prediction

**Model Sensors Used:**
- s2, s3, s4: Total temperature sensors (LPC, HPC, LPT)
- s7, s8, s9: Pressure sensors (HPC, HPT)
- s11, s12, s13: Flow sensors (physical fan/core speeds, corrected fan speed)
- s14, s15: Ratio and pressure ratio
- s17, s20, s21: Bleed enthalpy, HPC coolant, HPT coolant

### Classification Model (Placeholder)

The `is_going_to_fail` field is currently `null` because the classification model is not yet integrated. This will be added in a future update.

## Error Handling

All endpoints return standard HTTP status codes and error messages:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. This is an MVP deployment.

## Authentication

Currently, no authentication is required. For production deployment, implement:
- API key authentication
- JWT tokens
- Rate limiting per client

## Testing

Use the provided test script:

```bash
# Start the API
uvicorn app.main:app --reload

# Run tests (in another terminal)
python scripts/test_inference_api.py
```

## Examples

### Complete Example: Single Engine Prediction

```python
import requests
from datetime import datetime

# Prepare request
url = "http://localhost:8000/api/v1/inference/predict/batch"
data = {
    "batch_id": "test_001",
    "engines": [{
        "engine_id": "ENG-001",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": [
            {
                "unit": 1,
                "cycle": i,
                "setting_1": 0.0023,
                "setting_2": 0.0003,
                "setting_3": 100.0,
                "s2": 642.0,
                "s3": 1580.0,
                # ... other sensors
            }
            for i in range(128)
        ]
    }]
}

# Make request
response = requests.post(url, json=data)
result = response.json()

# Extract prediction
rul = result['predictions'][0]['remaining_useful_life']
print(f"Predicted RUL: {rul:.2f} cycles")
```

## Integration with Validation API

The inference API can be used together with the validation API:

1. **Make Prediction**: Use `/inference/predict/batch` to get RUL predictions
2. **Validate Data**: Use `/validate/batch` to check data quality and drift
3. **Monitor Model**: Track prediction stability over time using `/validate/summary`

This provides comprehensive monitoring of both data quality and model performance.

