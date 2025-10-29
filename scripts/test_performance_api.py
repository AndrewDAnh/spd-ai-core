"""
Test script for the performance evaluation API

Usage:
    python scripts/test_performance_api.py
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


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


def test_run_performance_evaluation():
    """Test running model performance evaluation"""
    print("\n" + "=" * 60)
    print("Testing Performance Evaluation (this may take a few minutes)")
    print("=" * 60)
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/models/performance/run",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Performance evaluation completed!")
            print(f"\nRegression Metrics:")
            print(f"  MSE:  {result['mean_squared_error']:.4f}")
            print(f"  MAE:  {result['mean_absolute_error']:.4f}")
            print(f"  MAPE: {result['mean_absolute_percentage_error']:.4f}")
            
            print(f"\nClassification Metrics:")
            if result['precision'] and result['recall']:
                print(f"  Precision (class 0): {result['precision'][0]:.4f}")
                print(f"  Precision (class 1): {result['precision'][1]:.4f}")
                print(f"  Recall (class 0):    {result['recall'][0]:.4f}")
                print(f"  Recall (class 1):    {result['recall'][1]:.4f}")
                print(f"  F1-score (macro):    {result['f1_score']:.4f}")
            else:
                print("  Classification model unavailable (placeholder metrics)")
            
            print(f"\nValidation Time: {result['validation_time']}")
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


def test_get_performance_metrics():
    """Test retrieving stored performance metrics"""
    print("\n" + "=" * 60)
    print("Testing Get Performance Metrics")
    print("=" * 60)
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/models/performance",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Retrieved performance metrics!")
            print(f"\nRegression Metrics:")
            print(f"  MSE:  {result['mean_squared_error']:.4f}")
            print(f"  MAE:  {result['mean_absolute_error']:.4f}")
            print(f"  MAPE: {result['mean_absolute_percentage_error']:.4f}")
            
            print(f"\nClassification Metrics:")
            if result['precision'] and result['recall']:
                print(f"  Precision (class 0): {result['precision'][0]:.4f}")
                print(f"  Precision (class 1): {result['precision'][1]:.4f}")
                print(f"  Recall (class 0):    {result['recall'][0]:.4f}")
                print(f"  Recall (class 1):    {result['recall'][1]:.4f}")
                print(f"  F1-score (macro):    {result['f1_score']:.4f}")
            else:
                print("  Classification model unavailable (placeholder metrics)")
            
            print(f"\nValidation Time: {result['validation_time']}")
            return True
        elif response.status_code == 404:
            print("\n✗ No performance metrics found. Run evaluation first.")
            return False
        else:
            print(f"\n✗ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to API")
        return False
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Model Performance API Test" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    
    # Try to get existing metrics first
    print("\n" + "-" * 60)
    print("Checking for existing performance metrics...")
    print("-" * 60)
    existing = test_get_performance_metrics()
    
    if not existing:
        print("\n" + "-" * 60)
        print("Running performance evaluation (this will take several minutes)...")
        print("-" * 60)
        results.append(("Run Performance Evaluation", test_run_performance_evaluation()))
        results.append(("Get Performance Metrics", test_get_performance_metrics()))
    else:
        results.append(("Get Performance Metrics", True))
        print("\n" + "-" * 60)
        print("Note: To re-run evaluation, call POST /models/performance/run")
        print("-" * 60)
    
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
