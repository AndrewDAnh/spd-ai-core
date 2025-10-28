# SPD-MVP: AI Agent Instructions

## Project Overview

FastAPI service for **jet engine asset management** providing RUL (Remaining Useful Life) prediction using the **STAR transformer model** and comprehensive data/model validation pipelines. Uses **C-MAPSS dataset format** (NASA turbofan degradation data) with 14 sensors and 3 operational settings.

## Architecture at a Glance

```
app/
├── api/endpoints/     # FastAPI routes: health, inference, validation
├── services/          # Business logic (inference, drift, quality, model drift)
├── db/               # SQLAlchemy models & CRUD (predictions, baselines)
├── models/schemas.py # Pydantic models (CmapssDataPoint, BatchPredictionRequest)
├── core/             # Config, database, logging
└── utils/            # CMAPSS loader, dataset utilities

scripts/              # ONE-TIME preprocessing (NOT production service)
models/regression/    # STAR model artifacts (config.yaml, checkpoints/best.pt)
```

**Key Separation**: `scripts/` contains data transformation/preprocessing logic (inject drift, outliers, missing values) that runs ONCE to create test datasets. NEVER add this to production service code in `app/`.

## Critical Patterns

### 1. C-MAPSS Data Format (NASA Turbofan Dataset)
All inference endpoints expect **C-MAPSS format** - never use generic sensor names without checking schema:

```python
# ✅ CORRECT - C-MAPSS format
class CmapssDataPoint(BaseModel):
    unit: int              # Engine identifier
    cycle: int             # Operational cycle (not datetime)
    setting_1: float       # Operational setting 1
    setting_2: float       # Operational setting 2  
    setting_3: float       # Operational setting 3
    s1: Optional[float]    # Sensor 1
    s2: Optional[float]    # Sensor 2
    # ... s3-s21 (all sensors optional)
```

**Features**: 3 operational settings + 21 sensors (24 total). The trained STAR model uses a subset of **14 sensors**: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21 (defined in `models/regression/fd001/normalisation.json`)

### 2. Model Architecture - STAR Regression Model
Production model in `app/services/star_model.py` (~490 lines):
- **Input**: 128 time steps × 14 sensors (subset selected during training)
- **Full C-MAPSS format**: 3 settings + 21 sensors (24 features total)
- **Used by model**: 14 sensors (s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21)
- **Output**: Single RUL value (float, cycles remaining)
- **Architecture**: Multi-scale hierarchical transformer with two-stage attention (temporal → sensor)
- **Loading**: `STARPredictionEngine(run_dir="models/regression/fd001", device="cuda|cpu")`
- **Preprocessing**: Interpolates missing values → median smoothing (window=5) → normalization

**Classification model** is NOT integrated - `is_going_to_fail` returns `None` (placeholder for future classifier).

### 3. Data Validation Pipeline Philosophy
Validation API **preserves missing values as `null`** to detect quality issues - NEVER fill/impute before validation:

```python
# ✅ CORRECT - Preserve None for quality checks
point[f"sensor_{i}"] = None if pd.isna(value) else float(value)

# ❌ WRONG - Hides data quality problems
point[f"sensor_{i}"] = median_value if pd.isna(value) else float(value)
```

See `docs/DESIGN_DECISIONS.md` for rationale.

### 4. Service Layer Pattern
All business logic in `app/services/` with singleton initialization in routers:

```python
# ✅ In router file (app/api/endpoints/inference.py)
model_service = ModelInferenceService()  # Singleton

@router.post("/predict/batch")
async def predict_batch(request, db):
    rul, is_fail, conf = model_service.predict(...)  # Use singleton
```

Services load heavy resources (models, configs) once at startup, not per-request.

### 5. Database Layer
SQLite with SQLAlchemy ORM - two main tables:
- **predictions**: Stores all RUL predictions (indexed by `engine_id`, `prediction_time`)
- **reference_baselines**: Stores drift detection baselines (one per engine, JSON blob)

CRUD operations in `app/db/crud.py` - always use provided functions, never raw SQL.

### 6. Drift Detection Methods
Three statistical tests for numerical features (in `app/services/drift_detector.py`):
- **KS Test**: Kolmogorov-Smirnov (distribution comparison)
- **PSI**: Population Stability Index (industry standard, primary metric)
- **Chi-square**: For categorical features

PSI thresholds: >0.2 = high drift, >0.1 = medium drift, ≤0.1 = no drift

### 7. Model Drift Detection
Tracks **prediction stability** over time (NOT model performance):

```python
# Formula
rul_change_rate = |RUL_current - RUL_previous| / hours_elapsed
```

Query consecutive predictions from DB, flag engines with high volatility (default threshold: 5.0).

## Development Workflows

### Running the API
```powershell
# Start server (default: http://localhost:8000)
uvicorn app.main:app --reload

# View docs
# http://localhost:8000/docs (Swagger)
# http://localhost:8000/redoc (ReDoc)
```

### Testing
```powershell
# Test inference endpoints
python scripts/test_inference_api.py

# Test validation endpoints  
python scripts/test_cmapss_validation.py

# Quick API check
python test_api.py
```

### Data Preprocessing (One-Time)
```powershell
# Generate test datasets with realistic issues (drift, missing, outliers)
python scripts/preprocess_cmapss.py

# Output: examples/cmapss/*.json
```

**Never** import `scripts/realistic_transform.py` into `app/` - transformations are offline preprocessing only.

### Configuration
Edit `.env` or set environment variables:
```bash
REGRESSION_MODEL_PATH=models/regression/fd001  # STAR model location
DEVICE=cuda                                    # or 'cpu'
DRIFT_THRESHOLD=0.2                           # PSI threshold
FAILURE_THRESHOLD=30                          # RUL warning level
```

Settings loaded via `app/core/config.py` using Pydantic with `@lru_cache()`.

## Common Mistakes to Avoid

1. **Don't modify API schemas to accept generic sensor names** - preprocessing scripts handle C-MAPSS → API format conversion
2. **Don't add data transformation logic to `app/services/`** - keep in `scripts/` for offline use
3. **Don't map cycles to datetime** - aviation domain uses operational cycles, not wall-clock time
4. **Don't fill missing values in validation requests** - defeats purpose of quality monitoring
5. **Don't create new database queries** - use CRUD functions in `app/db/crud.py`
6. **Don't initialize services in endpoint functions** - use module-level singletons

## Key Files Reference

| File | Purpose |
|------|---------|
| `app/services/star_model.py` | STAR model architecture + prediction engine |
| `app/services/model_inference.py` | Inference service (loads STAR, manages predictions) |
| `app/services/drift_detector.py` | KS/PSI/Chi-square statistical tests |
| `app/services/model_drift.py` | Prediction stability tracking |
| `app/models/schemas.py` | Pydantic schemas (CmapssDataPoint is critical) |
| `app/db/crud.py` | Database operations (predictions, baselines) |
| `docs/DESIGN_DECISIONS.md` | Design rationale (missing values, transformations) |
| `STAR_MODEL_INTEGRATION.md` | Model integration details |

## Production Notes

- **SQLite** used for MVP - migrate to PostgreSQL for production
- **No authentication** - add JWT/OAuth2 for production
- **Mock confidence scores** - replace with proper uncertainty estimates when available
- **Classification model pending** - `is_going_to_fail` returns `null` until integrated

## Testing Conventions

Example requests in `examples/`:
- `predict_request.json` - Batch prediction
- `validation_request.json` - Batch validation
- `examples/cmapss/*.json` - Preprocessed C-MAPSS data

Always test with 128 time steps for optimal STAR model performance (minimum length for accuracy).
