# Design Decisions

## 1. Handling Missing Values in Validation API

### Decision
**Preserve missing values as `null` in the API input, rather than filling them.**

### Rationale
The primary purpose of the validation API is to detect data quality issues, including missing values. If we fill missing values before sending them to the API:

- ❌ Quality checks cannot detect missing data
- ❌ Hides real production issues
- ❌ False sense of data completeness
- ❌ Cannot distinguish between real and imputed values

By preserving `null` values:

- ✅ Quality checks can detect and report missing values
- ✅ Accurate representation of production data
- ✅ Enables proper monitoring and alerting
- ✅ Maintains data integrity throughout the pipeline

### Implementation

**API Schemas** (`app/models/schemas.py`):
```python
class EngineValidationData(BaseModel):
    engine_id: str
    data: List[Dict[str, Optional[float]]]  # Allow None for missing values
```

**Format Converter** (`scripts/format_converter.py`):
```python
# Preserve NaN as None (null in JSON) for quality checks
if pd.isna(value):
    point[f"sensor_{i}"] = None
else:
    point[f"sensor_{i}"] = float(value)
```

**Validation Service** (`app/services/validation_service.py`):
- Quality checks detect `None` values
- Missing values flagged as high severity issues
- Reported in validation results

### Test Results

With ~10% missing values injected in test data:

```
Batch Validation Results:
  - High severity: 28 (missing values detected)
  - Medium severity: 91 (other quality issues)
  - Total quality issues: 238
```

### Alternative Considered

**Fill missing values with medians/means:**
- Pros: Simplifies downstream processing
- Cons: Defeats the purpose of validation, hides issues
- **Rejected:** Not suitable for a validation/monitoring system

---

## 2. Separate Data Transformation Scripts

### Decision
**Keep data transformation logic (injecting missing values, outliers, drift) separate from the production validation service.**

### Rationale

**One-Time Preprocessing:**
- Data transformations only needed once to create realistic test datasets
- Not part of the production service runtime
- Avoids unnecessary complexity in API code

**Separation of Concerns:**
- `scripts/` - Preprocessing and test data generation
- `app/` - Production validation service
- Clear boundary between test data preparation and production logic

### Implementation

**Preprocessing Scripts:**
- `scripts/realistic_transform.py` - Transformation functions
- `scripts/preprocess_cmapss.py` - One-time pipeline
- `scripts/format_converter.py` - Format conversion

**Production Service:**
- `app/services/validation_service.py` - Only validation logic
- No data transformation code
- Receives pre-formatted JSON

---

## 3. C-MAPSS Format Compatibility

### Decision
**Map C-MAPSS sensor names to API format in preprocessing, rather than modifying the API.**

### Rationale

**Keep API Generic:**
- Current format (`sensor_1`, `sensor_2`, ...) is dataset-agnostic
- Works for any sensor-based monitoring system
- No coupling to specific dataset formats

**Preprocessing Handles Mapping:**
- `s1` → `sensor_1`, `s2` → `sensor_2`, etc.
- Engine IDs: `1` → `ENG-001`, `2` → `ENG-002`, etc.
- Format conversion happens once during preprocessing

**Benefits:**
- No API changes needed
- Easy to integrate other datasets later
- Clear separation: preprocessing vs. validation

### Alternative Considered

**Add C-MAPSS-specific endpoint:**
- Pros: More flexible for multiple formats
- Cons: Increases API complexity, couples to specific datasets
- **Rejected:** Preprocessing approach is cleaner and simpler

---

## 4. Time Representation

### Decision
**Keep cycle numbers as-is, do not map to datetime.**

### Rationale

**Aviation Context:**
- Engines tracked by operational cycles (flight hours, starts)
- Cycle numbers are the natural time unit
- RUL predictions based on cycles, not wall-clock time

**Simplicity:**
- No arbitrary datetime mapping needed
- Avoids introducing artificial temporal patterns
- More accurate for aviation use case

**MVP Scope:**
- Datetime mapping can be added later if needed
- Current approach sufficient for validation MVP
- Future inference pipeline can handle time conversion if required

---

## 5. Realistic Data Transformation Rates

### Decision
**Use conservative transformation rates that reflect real production quality.**

### Rationale

**Initial Problem:**
- Original rates (10% missing, 25% drift, 8% anomalous, 2x noise)
- Resulted in ~24 issues per engine
- Unrealistic for well-maintained aviation systems

**Production Reality:**
- Aviation-grade sensors are highly reliable
- Well-maintained systems have low failure rates
- Issues should be exceptions, not the norm

### Implementation

**Current Configuration** (`scripts/realistic_transform.py`):
```python
TRANSFORM_CONFIG = {
    'missing_rate': 0.03,      # 3% missing values
    'outlier_rate': 0.03,      # 3% outliers
    'drift_ratio': 0.12,       # 12% engines with drift
    'noise_multiplier': 0.0,   # No additional noise (NASA data is good)
    'anomalous_ratio': 0.04,   # 4% anomalous engines
    'random_seed': 42
}
```

**Results:**
- ~6 issues per engine on average
- High severity: 12 issues (missing values)
- Medium severity: 51 issues (outliers, drift)
- 10% of engines flagged (1 out of 10)

### Benefits

**Realistic Testing:**
- Matches expected quality in production
- Appropriate for aviation domain
- Tests all quality checks effectively

**Better Signal-to-Noise:**
- Issues are meaningful and actionable
- No alarm fatigue from overwhelming issues
- Clear distinction between normal and problematic engines

**Reference Baseline:**
- Clean reference data (no transformations)
- Pristine baseline for drift detection
- Matches best practices for baseline establishment

### Documentation

See [TRANSFORMATION_RATES_COMPARISON.md](./TRANSFORMATION_RATES_COMPARISON.md) for:
- Detailed before/after comparison
- Impact analysis
- Justification for each rate

---

## Summary

These design decisions prioritize:
1. **Data Integrity** - Preserve real data issues for detection
2. **Simplicity** - Keep MVP focused and maintainable
3. **Separation of Concerns** - Clear boundaries between components
4. **Flexibility** - Generic API that works with various datasets
5. **Realism** - Transformation rates that reflect production quality

