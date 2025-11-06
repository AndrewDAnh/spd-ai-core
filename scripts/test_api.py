"""
Simple test script to verify the API is working correctly.
This script tests all major endpoints.

Usage:
    python test_api.py
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Health check passed")


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n=== Testing Batch Prediction ===")
    
    with open("examples/predict_request.json", "r") as f:
        payload = json.load(f)
    
    response = requests.post(f"{BASE_URL}/api/v1/predict/batch", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Batch prediction passed")
    
    return response.json()


def test_prediction_history(engine_id):
    """Test prediction history endpoint"""
    print(f"\n=== Testing Prediction History for {engine_id} ===")
    
    response = requests.get(f"{BASE_URL}/api/v1/predict/history/{engine_id}?limit=10")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Prediction history passed")


def test_store_reference():
    """Test storing reference baseline"""
    print("\n=== Testing Store Reference ===")
    
    with open("examples/reference_request.json", "r") as f:
        payload = json.load(f)
    
    response = requests.post(f"{BASE_URL}/api/v1/validate/reference", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Store reference passed")


def test_batch_validation():
    """Test batch validation endpoint"""
    print("\n=== Testing Batch Validation ===")
    
    with open("examples/validation_request.json", "r") as f:
        payload = json.load(f)
    
    response = requests.post(f"{BASE_URL}/api/v1/validate/batch", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Batch validation passed")


def test_drift_detection():
    """Test drift detection endpoint"""
    print("\n=== Testing Drift Detection ===")
    
    with open("examples/validation_request.json", "r") as f:
        payload = json.load(f)
    
    response = requests.post(f"{BASE_URL}/api/v1/validate/drift", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Drift detection passed")


def test_quality_check():
    """Test quality check endpoint"""
    print("\n=== Testing Quality Check ===")
    
    with open("examples/validation_request.json", "r") as f:
        payload = json.load(f)
    
    response = requests.post(f"{BASE_URL}/api/v1/validate/quality", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Quality check passed")


def test_model_drift():
    """Test model drift detection endpoint"""
    print("\n=== Testing Model Drift Detection ===")
    
    with open("examples/model_drift_request.json", "r") as f:
        payload = json.load(f)
    
    response = requests.post(f"{BASE_URL}/api/v1/validate/model-drift", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Model drift detection passed")


def test_validation_summary():
    """Test validation summary endpoint"""
    print("\n=== Testing Validation Summary ===")
    
    response = requests.get(f"{BASE_URL}/api/v1/validate/summary")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Validation summary passed")


def main():
    """Run all tests"""
    print("=" * 60)
    print("SPD-MVP API Test Suite")
    print("=" * 60)
    
    try:
        # Test health check
        test_health()
        
        # Test prediction endpoints
        prediction_result = test_batch_prediction()
        engine_id = prediction_result["predictions"][0]["engine_id"]
        test_prediction_history(engine_id)
        
        # Test validation endpoints
        test_store_reference()
        test_batch_validation()
        test_drift_detection()
        test_quality_check()
        test_model_drift()
        test_validation_summary()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running: uvicorn app.main:app --reload")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    main()

