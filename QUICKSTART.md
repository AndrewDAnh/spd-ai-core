# Quick Start Guide - 5 Minutes to Running

## Step 1: Setup (2 minutes)

```bash
# Navigate to project
cd spd-mvp

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

## Step 2: Start the API (30 seconds)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

✓ API running at: http://localhost:8000
✓ Documentation: http://localhost:8000/docs

## Step 3: Test with Examples (2 minutes)

Open a new terminal and run:

```bash
# Activate the same virtual environment
source venv/bin/activate

# Run test suite
python test_api.py
```

You should see all tests passing! ✓

## Step 4: Try Your First Request

### Make a Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

Expected response:
```json
{
  "prediction_id": "pred_abc123",
  "batch_id": "batch_20250125_001",
  "predictions": [
    {
      "engine_id": "ENG-001",
      "remaining_useful_life": 145.5,
      "is_going_to_fail": false,
      "confidence": 0.87
    }
  ]
}
```

### Store Reference Baseline

```bash
curl -X POST "http://localhost:8000/api/v1/validate/reference" \
  -H "Content-Type: application/json" \
  -d @examples/reference_request.json
```

### Validate Data

```bash
curl -X POST "http://localhost:8000/api/v1/validate/batch" \
  -H "Content-Type: application/json" \
  -d @examples/validation_request.json
```

## Step 5: Explore the API

Visit http://localhost:8000/docs for:
- Interactive API documentation
- Try all endpoints in your browser
- See request/response schemas

## Common Commands

### Start API
```bash
uvicorn app.main:app --reload
```

### Start in Background
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

### Check API Status
```bash
curl http://localhost:8000/api/v1/health
```

### View Logs
```bash
tail -f api.log
```

### Stop API (if running in background)
```bash
ps aux | grep uvicorn
kill <PID>
```

## Project Structure

```
spd-mvp/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/endpoints/       # API routes
│   ├── services/            # Business logic
│   ├── db/                  # Database models
│   └── models/schemas.py    # Request/response models
├── examples/                # Sample requests
├── data/                    # SQLite database
├── README.md               # Full documentation
├── QUICKSTART.md           # This file
└── requirements.txt        # Dependencies
```

## Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/predict/batch` | POST | Batch predictions |
| `/api/v1/predict/history/{id}` | GET | Prediction history |
| `/api/v1/validate/batch` | POST | Full validation |
| `/api/v1/validate/drift` | POST | Drift detection |
| `/api/v1/validate/quality` | POST | Quality checks |
| `/api/v1/validate/model-drift` | POST | Model drift |
| `/api/v1/validate/reference` | POST | Store baseline |
| `/api/v1/validate/summary` | GET | Validation metrics |

## Using Your Own Data

### 1. Prediction Request Format

```json
{
  "batch_id": "your_batch_id",
  "engines": [
    {
      "engine_id": "YOUR-ENGINE-ID",
      "timestamp": "2025-01-25T10:30:00Z",
      "data": [
        {"sensor_1": 518.67, "sensor_2": 641.82, "...": "..."}
      ]
    }
  ]
}
```

### 2. Validation Request Format

```json
{
  "validation_id": "your_validation_id",
  "engines": [
    {
      "engine_id": "YOUR-ENGINE-ID",
      "data": [
        {"sensor_1": 525.20, "sensor_2": 650.50, "...": "..."}
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

## Troubleshooting

### Port Already in Use
```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>
```

### Import Errors
```bash
# Make sure you're in the right directory
cd spd-mvp
# And virtual environment is activated
source venv/bin/activate
```

### Database Issues
```bash
# Remove and recreate database
rm data/predictions.db
# Restart the API - database will be recreated
```

## Next Steps

1. ✓ Read [README.md](README.md) for detailed documentation
2. ✓ Review [CLIENT_PROPOSAL.md](CLIENT_PROPOSAL.md) for business context
3. ✓ Check [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
4. ✓ Customize the API for your specific needs
5. ✓ Integrate with your existing systems

## Support

- Documentation: http://localhost:8000/docs
- Test Suite: `python test_api.py`
- Example Requests: `examples/` directory

**You're ready to go! 🚀**

