# SPD-MVP: AI Model Validation & Inference API

FastAPI service for jet engine asset management providing:
- **STAR Model** inference pipeline (RUL prediction using Spatio-Temporal Attention)
- Data validation (drift detection, quality checks)
- Model drift monitoring (prediction stability tracking)

## Features

### Inference Pipeline ⭐ NEW
- **STAR Regression Model**: State-of-the-art transformer architecture for RUL prediction
- Batch predictions for multiple engines
- C-MAPSS data format support (NASA turbofan dataset)
- RUL (Remaining Useful Life) prediction with confidence scores
- Failure prediction placeholder (classifier to be integrated)
- Prediction history storage in SQLite

### Validation Pipeline
- **Data Drift Detection**: KS test, PSI, Chi-square for distribution changes
- **Data Quality Checks**: Missing values, outliers, schema validation
- **Model Drift Detection**: Prediction stability tracking over time

## Quick Start

### Installation

```bash
# Clone repository
cd spd-mvp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### Using C-MAPSS Dataset

The API uses the C-MAPSS dataset format (NASA turbofan engine data):

```bash
# Preprocess C-MAPSS validation data
python3 scripts/preprocess_cmapss.py

# Start API
uvicorn app.main:app --reload

# Test inference API
python3 scripts/test_inference_api.py

# Test validation API
python3 scripts/test_cmapss_validation.py
```

See [docs/CMAPSS_INTEGRATION.md](docs/CMAPSS_INTEGRATION.md) for validation details and [docs/API.md](docs/API.md) for inference API documentation.

### Run the API

```bash
# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

- **API Documentation**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
GET /api/v1/health
```

### Inference Endpoints

#### Batch Prediction
```bash
POST /api/v1/predict/batch
```

**Request (C-MAPSS Format):**
```json
{
  "batch_id": "batch_20250125_001",
  "engines": [
    {
      "engine_id": "ENG-001",
      "timestamp": "2025-01-25T10:30:00Z",
      "data": [
        {
          "unit": 1,
          "cycle": 1,
          "setting_1": 0.0023,
          "setting_2": 0.0003,
          "setting_3": 100.0,
          "s2": 641.82,
          "s3": 1589.7,
          "s4": 1400.60,
          "s7": 554.36,
          "s8": 2388.06,
          "s9": 9046.19,
          "s11": 47.47,
          "s12": 521.66,
          "s13": 2388.02,
          "s14": 8138.62,
          "s15": 8.41,
          "s17": 392.00,
          "s20": 39.06,
          "s21": 23.42
        }
      ]
    }
  ]
}
```

Note: The API now uses C-MAPSS format with `unit`, `cycle`, `setting_1-3`, and sensors `s1-s21` (sensors are optional)

**Response:**
```json
{
  "prediction_id": "pred_abc123",
  "batch_id": "batch_20250125_001",
  "timestamp": "2025-01-25T10:30:05Z",
  "predictions": [
    {
      "engine_id": "ENG-001",
      "prediction_time": "2025-01-25T10:30:05Z",
      "remaining_useful_life": 85.34,
      "is_going_to_fail": null,
      "confidence": 0.87
    }
  ]
}
```

Note: `is_going_to_fail` is currently `null` as the classification model is not yet integrated

#### Get Prediction History
```bash
GET /api/v1/predict/history/{engine_id}?limit=100
```

### Validation Endpoints

#### Store Reference Baseline
```bash
POST /api/v1/validate/reference
```

**Request:**
```json
{
  "engine_id": "ENG-001",
  "reference_data": [
    {"sensor_1": 518.5, "sensor_2": 642.0, "sensor_3": 1590.0},
    {"sensor_1": 518.6, "sensor_2": 642.2, "sensor_3": 1590.5}
  ]
}
```

#### Batch Validation (Drift + Quality)
```bash
POST /api/v1/validate/batch
```

**Request:**
```json
{
  "validation_id": "val_20250125_001",
  "engines": [
    {
      "engine_id": "ENG-001",
      "data": [
        {"sensor_1": 525.2, "sensor_2": 650.5, "sensor_3": 1600.0}
      ]
    }
  ],
  "use_stored_reference": true,
  "config": {
    "drift_threshold": 0.2,
    "outlier_sensitivity": "medium"
  }
}
```

**Response:**
```json
{
  "validation_id": "val_20250125_001",
  "timestamp": "2025-01-25T10:35:00Z",
  "summary": {
    "total_engines": 1,
    "engines_with_issues": 0,
    "high_severity_count": 0,
    "medium_severity_count": 0,
    "drift_detected_count": 0
  },
  "engines": [
    {
      "engine_id": "ENG-001",
      "status": "ok",
      "drift_detected": false,
      "quality_passed": true
    }
  ]
}
```

#### Data Drift Detection Only
```bash
POST /api/v1/validate/drift
```

#### Data Quality Check Only
```bash
POST /api/v1/validate/quality
```

#### Model Drift Detection
```bash
POST /api/v1/validate/model-drift
```

**Request:**
```json
{
  "engines": ["ENG-001", "ENG-002"],
  "lookback_hours": 24,
  "threshold": 5.0
}
```

**Response:**
```json
{
  "timestamp": "2025-01-25T10:35:00Z",
  "summary": {
    "total_engines": 2,
    "unstable_engines": 0,
    "avg_stability_score": 2.1
  },
  "engines": [
    {
      "engine_id": "ENG-001",
      "status": "stable",
      "prediction_count": 48,
      "avg_rul_change_rate": 2.1,
      "max_rul_change_rate": 4.5,
      "consecutive_predictions": [
        {
          "time": "2025-01-25T08:00:00Z",
          "rul": 145.5,
          "change_rate": null
        },
        {
          "time": "2025-01-25T09:00:00Z",
          "rul": 143.2,
          "change_rate": 2.3
        }
      ]
    }
  ]
}
```

#### Validation Summary
```bash
GET /api/v1/validate/summary
```

## Example Usage with cURL

### Make a Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

### Validate Data
```bash
# First, store reference baseline
curl -X POST "http://localhost:8000/api/v1/validate/reference" \
  -H "Content-Type: application/json" \
  -d @examples/reference_request.json

# Then validate current data
curl -X POST "http://localhost:8000/api/v1/validate/batch" \
  -H "Content-Type: application/json" \
  -d @examples/validation_request.json
```

### Check Model Drift
```bash
curl -X POST "http://localhost:8000/api/v1/validate/model-drift" \
  -H "Content-Type: application/json" \
  -d '{
    "engines": ["ENG-001", "ENG-002"],
    "lookback_hours": 24,
    "threshold": 5.0
  }'
```

## Configuration

Edit `.env` file to configure:

```bash
# Application
APP_NAME=SPD-MVP
DEBUG=True

# Database
DATABASE_URL=sqlite:///./data/predictions.db

# Model Settings - STAR Regression Model
REGRESSION_MODEL_PATH=models/regression/fd001
DEVICE=cuda  # or 'cpu'
FAILURE_THRESHOLD=30

# Validation
DRIFT_THRESHOLD=0.2
PSI_THRESHOLD=0.2
OUTLIER_SENSITIVITY=medium  # low, medium, high
```

## Architecture

```
spd-mvp/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── api/endpoints/          # API route handlers
│   ├── core/                   # Config, database, logging
│   ├── db/                     # Database models & CRUD
│   ├── services/               # Business logic
│   │   ├── preprocessor.py
│   │   ├── model_inference.py
│   │   ├── drift_detector.py
│   │   ├── quality_checker.py
│   │   └── model_drift.py
│   ├── models/schemas.py       # Pydantic models
│   └── utils/                  # Utilities
└── data/                       # SQLite database
```

## Data Flow

### Inference Pipeline
1. Client sends time-series data
2. Data preprocessing (cleaning, scaling)
3. Model inference (RUL prediction)
4. Store predictions in SQLite
5. Return predictions with confidence

### Validation Pipeline
1. Client sends current data + reference baseline
2. **Drift Detection**: Statistical tests (KS, PSI, Chi-square)
3. **Quality Checks**: Missing values, outliers, schema
4. **Model Drift**: Query prediction history, calculate stability
5. Return validation results with severity levels

## Drift Detection Methods

### Data Drift
- **KS Test**: Kolmogorov-Smirnov test for numerical distributions
- **PSI**: Population Stability Index for binned distributions
- **Chi-square**: Chi-square test for categorical features

### Model Drift
- **Prediction Stability**: `|RUL_current - RUL_previous| / hours_elapsed`
- Tracks prediction volatility over time
- Flags engines with high change rates

## Model Architecture

### STAR Regression Model

The inference API uses the **STAR** (Spatio-Temporal Attention for RUL) model:

- **Architecture**: Multi-scale transformer with two-stage attention (temporal + sensor)
- **Input**: 128 time steps × 14 sensors
- **Output**: RUL prediction (cycles remaining)
- **Training Dataset**: C-MAPSS FD001 (NASA turbofan degradation)
- **Performance**: State-of-the-art accuracy on turbofan RUL prediction

**Key Features:**
- Patch-based embedding for efficient processing
- Hierarchical encoder-decoder with 3 scales
- Two-stage attention: temporal (within sensors) + sensor (across sensors)
- Pre-trained on NASA C-MAPSS FD001 dataset

### Classification Model (Placeholder)

The failure classification model is not yet integrated. The API returns `null` for `is_going_to_fail` until the classifier is available.

## MVP Status

Current implementation:
- ✅ **STAR Regression Model**: Fully integrated and operational
- ✅ **C-MAPSS Data Format**: Native support
- ✅ **Batch Inference**: Multiple engines in single request
- ✅ **Validation Pipeline**: Drift detection and quality checks
- ✅ **Model Drift Monitoring**: Prediction stability tracking
- ⏳ **Classification Model**: Placeholder (to be integrated)
- ❌ **Authentication**: Not implemented (add for production)
- ❌ **Comprehensive Tests**: Limited test coverage

## Production Recommendations

For production deployment:
1. ✅ ~~Replace mock model with trained model~~ (DONE - STAR model integrated)
2. Integrate classification model for `is_going_to_fail` prediction
3. Add authentication (JWT, OAuth2)
4. Add comprehensive test suite
5. Use PostgreSQL instead of SQLite
6. Add monitoring and alerting
7. Implement rate limiting
8. Add request validation middleware
9. Deploy with Docker/Kubernetes
10. Add CI/CD pipeline
11. Implement proper logging aggregation

## License

Proprietary - Client Project

## Contact

For questions about this MVP, contact the development team.

