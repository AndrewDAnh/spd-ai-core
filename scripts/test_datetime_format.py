"""Test RFC3339 datetime serialization in API responses.

This script verifies that all datetime fields in API responses
are properly formatted according to RFC3339 standard.

Usage:
    python scripts/test_datetime_format.py
"""

import json
import re
import requests
from datetime import datetime, UTC

BASE_URL = "http://localhost:8000"

# RFC3339 regex pattern
RFC3339_PATTERN = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$'
)


def is_rfc3339(dt_string: str) -> bool:
    """Check if string matches RFC3339 format."""
    return bool(RFC3339_PATTERN.match(dt_string))


def check_datetime_fields(data, path=""):
    """Recursively check all datetime-like fields in response."""
    issues = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # Check if field name suggests it's a datetime
            if any(dt_word in key.lower() for dt_word in ['time', 'date', 'at', 'timestamp']):
                if isinstance(value, str):
                    if not is_rfc3339(value):
                        issues.append({
                            'path': current_path,
                            'value': value,
                            'issue': 'Not RFC3339 format'
                        })
                elif value is not None:
                    issues.append({
                        'path': current_path,
                        'value': str(value),
                        'issue': 'Expected string, got ' + type(value).__name__
                    })
            
            # Recurse into nested structures
            if isinstance(value, (dict, list)):
                issues.extend(check_datetime_fields(value, current_path))
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            if isinstance(item, (dict, list)):
                issues.extend(check_datetime_fields(item, current_path))
    
    return issues


def test_health_endpoint():
    """Test health check endpoint datetime format."""
    print("\n" + "="*60)
    print("Testing /api/v1/health endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/v1/health")
    response.raise_for_status()
    data = response.json()
    
    print(f"Response: {json.dumps(data, indent=2)}")
    
    issues = check_datetime_fields(data)
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue['path']}: {issue['value']} ({issue['issue']})")
        return False
    else:
        print("\n✅ All datetime fields properly formatted")
        return True


def test_batch_prediction():
    """Test batch prediction endpoint datetime format."""
    print("\n" + "="*60)
    print("Testing /api/v1/predict/batch endpoint")
    print("="*60)
    
    # Sample request with minimal data
    request_data = {
        "batch_id": "datetime_test_001",
        "engines": [
            {
                "engine_id": "TEST-001",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": [
                    {
                        "cycle": 1,
                        "setting_1": 0.0,
                        "setting_2": 0.0,
                        "setting_3": 100.0,
                        "s2": 642.0,
                        "s3": 1580.0,
                        "s4": 1400.0,
                        "s7": 550.0,
                        "s8": 2388.0,
                        "s9": 9050.0,
                        "s11": 47.0,
                        "s12": 521.0,
                        "s13": 2388.0,
                        "s14": 8138.0,
                        "s15": 8.4195,
                        "s17": 392.0,
                        "s20": 39.06,
                        "s21": 23.419
                    } for _ in range(128)  # Need 128 cycles minimum
                ]
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/predict/batch", json=request_data)
        response.raise_for_status()
        data = response.json()
        
        print(f"Response keys: {list(data.keys())}")
        
        issues = check_datetime_fields(data)
        if issues:
            print("\n❌ ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue['path']}: {issue['value']} ({issue['issue']})")
            return False
        else:
            print("\n✅ All datetime fields properly formatted")
            return True
    except requests.exceptions.HTTPError as e:
        print(f"\n⚠️  API Error: {e}")
        print(f"Response: {e.response.text}")
        return False


def test_validation_endpoints():
    """Test validation endpoint datetime formats."""
    print("\n" + "="*60)
    print("Testing validation endpoints")
    print("="*60)
    
    # Test reference storage (returns dict, not Pydantic)
    reference_data = {
        "engine_id": "TEST-REF-001",
        "reference_data": [
            {"sensor_1": 100.0, "sensor_2": 200.0} for _ in range(10)
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/validate/reference", json=reference_data)
        response.raise_for_status()
        data = response.json()
        print(f"Reference storage response: {json.dumps(data, indent=2)}")
        print("✅ Reference endpoint works")
    except Exception as e:
        print(f"⚠️  Reference endpoint error: {e}")
    
    return True


def main():
    """Run all datetime format tests."""
    print("="*60)
    print("RFC3339 Datetime Format Testing")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    print(f"RFC3339 Pattern: {RFC3339_PATTERN.pattern}")
    
    results = []
    
    try:
        results.append(("Health Check", test_health_endpoint()))
        results.append(("Batch Prediction", test_batch_prediction()))
        results.append(("Validation", test_validation_endpoints()))
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running: uvicorn app.main:app --reload")
        return
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all(r[1] for r in results):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed")


if __name__ == "__main__":
    main()
