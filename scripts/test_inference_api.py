"""
Test script for the inference API with STAR model integration

Usage:
    python scripts/test_inference_api.py
"""

import requests
import json
from datetime import datetime, UTC

BASE_URL = "http://localhost:8000"


def test_batch_prediction():
    """Test batch prediction with C-MAPSS format data"""
    print("=" * 60)
    print("Testing Batch Prediction API")
    print("=" * 60)
    
    # Example request with C-MAPSS format
    request_data = {
        "batch_id": "test_batch_001",
        "engines": [
            {
                "engine_id": "ENG-001",
                "timestamp": datetime.now(UTC).isoformat(),
                "data": [
                    {
                        "unit": 1,
                        "cycle": i,
                        "setting_1": 0.0023,
                        "setting_2": 0.0003,
                        "setting_3": 100.0,
                        "s2": 642.0 + i * 0.1,
                        "s3": 1580.0 + i * 0.2,
                        "s4": 1400.0 + i * 0.15,
                        "s7": 550.0 + i * 0.05,
                        "s8": 2388.0 + i * 0.1,
                        "s9": 9050.0 + i * 0.5,
                        "s11": 47.0 + i * 0.01,
                        "s12": 520.0 + i * 0.05,
                        "s13": 2388.0 + i * 0.1,
                        "s14": 8150.0 + i * 0.3,
                        "s15": 8.4 + i * 0.001,
                        "s17": 390.0 + i * 0.02,
                        "s20": 38.0 + i * 0.005,
                        "s21": 23.0 + i * 0.003,
                    }
                    for i in range(128)  # 128 time steps (sequence length)
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/batch",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Prediction successful!")
            print(f"\nPrediction ID: {result['prediction_id']}")
            print(f"Batch ID: {result['batch_id']}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"\nPredictions:")
            for pred in result['predictions']:
                print(f"  Engine: {pred['engine_id']}")
                print(f"    RUL: {pred['remaining_useful_life']:.2f} cycles")
                print(f"    Is Going to Fail: {pred['is_going_to_fail']}")
                print(f"    Confidence: {pred['confidence']:.2f}")
            return True
        else:
            print(f"\n✗ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to API. Make sure it's running:")
        print("  uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        if response.status_code == 200:
            print("\n✓ API is healthy")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"\n✗ Health check failed: {response.status_code}")
            return False
    except:
        print("\n✗ Could not reach API")
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "STAR Model Inference API Test" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("Batch Prediction", test_batch_prediction()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    print(f"\n  Total: {passed_count}/{len(results)} tests passed")
    
    if all(passed for _, passed in results):
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

