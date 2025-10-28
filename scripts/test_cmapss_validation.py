"""
Test validation API with C-MAPSS preprocessed data

Usage:
    python scripts/test_cmapss_validation.py
    
Prerequisites:
    1. Run scripts/preprocess_cmapss.py to generate data
    2. Start API: uvicorn app.main:app --reload
"""

import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://localhost:55005"


def load_json(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath) as f:
        return json.load(f)


def test_health_check():
    """Test API health"""
    print("\n" + "=" * 60)
    print("Test 1: Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"✓ API is healthy")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"✗ Health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to API at {BASE_URL}")
        print(f"  Make sure the API is running: uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_store_reference():
    """Test storing reference baselines"""
    print("\n" + "=" * 60)
    print("Test 2: Store Reference Baselines")
    print("=" * 60)
    
    try:
        reference = load_json("examples/cmapss/reference_baseline.json")
        
        # Store first 5 engines as reference
        success_count = 0
        for i in range(min(5, len(reference["engines"]))):
            engine = reference["engines"][i]
            ref_request = {
                "engine_id": engine["engine_id"],
                "reference_data": engine["data"]
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v1/validate/reference",
                json=ref_request,
                timeout=30
            )
            
            if response.status_code == 200:
                success_count += 1
                print(f"  ✓ Stored reference for {engine['engine_id']}")
            else:
                print(f"  ✗ Failed to store {engine['engine_id']}: {response.status_code}")
        
        print(f"\nStored {success_count}/5 reference baselines")
        return success_count > 0
        
    except FileNotFoundError:
        print(f"✗ Reference data not found. Run scripts/preprocess_cmapss.py first")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_batch_validation():
    """Test batch validation endpoint"""
    print("\n" + "=" * 60)
    print("Test 3: Batch Validation (Drift + Quality)")
    print("=" * 60)
    
    try:
        current = load_json("examples/cmapss/current_data.json")
        
        # Test with first 10 engines
        val_request = {
            "validation_id": "cmapss_test_001",
            "engines": current["engines"][:10],
            "use_stored_reference": True,
            "config": {
                "drift_threshold": 0.2,
                "outlier_sensitivity": "medium"
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/validate/batch",
            json=val_request,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Validation completed")
            print(f"\n  Summary:")
            print(f"    - Total engines: {result['summary']['total_engines']}")
            print(f"    - Engines with issues: {result['summary']['engines_with_issues']}")
            print(f"    - Drift detected: {result['summary']['drift_detected_count']}")
            print(f"    - High severity: {result['summary']['high_severity_count']}")
            print(f"    - Medium severity: {result['summary']['medium_severity_count']}")
            
            # Show first engine with issues
            for engine in result['engines']:
                if engine['status'] != 'ok':
                    print(f"\n  Example - {engine['engine_id']}:")
                    print(f"    Status: {engine['status']}")
                    print(f"    Drift: {engine['drift_detected']}")
                    print(f"    Quality: {engine['quality_passed']}")
                    if engine.get('issues'):
                        print(f"    Issues: {len(engine['issues'])}")
                        for issue in engine['issues'][:3]:
                            print(f"      - {issue['feature']}: {issue['type']} ({issue['severity']})")
                    break
            
            return True
        else:
            print(f"✗ Validation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drift_detection():
    """Test drift detection only"""
    print("\n" + "=" * 60)
    print("Test 4: Drift Detection Only")
    print("=" * 60)
    
    try:
        current = load_json("examples/cmapss/current_data.json")
        
        val_request = {
            "validation_id": "cmapss_drift_test",
            "engines": current["engines"][:5],
            "use_stored_reference": True,
            "config": {
                "drift_threshold": 0.2
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/validate/drift",
            json=val_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Drift detection completed")
            print(f"  Results for {len(result['results'])} engines")
            
            drift_count = sum(1 for r in result['results'] if r.get('drift_detected', False))
            print(f"  Engines with drift: {drift_count}/{len(result['results'])}")
            
            return True
        else:
            print(f"✗ Drift detection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_quality_check():
    """Test quality check only"""
    print("\n" + "=" * 60)
    print("Test 5: Quality Check Only")
    print("=" * 60)
    
    try:
        current = load_json("examples/cmapss/current_data.json")
        
        val_request = {
            "validation_id": "cmapss_quality_test",
            "engines": current["engines"][:5],
            "config": {
                "outlier_sensitivity": "medium"
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/validate/quality",
            json=val_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Quality check completed")
            print(f"  Results for {len(result['results'])} engines")
            
            issues_count = sum(len(r.get('issues', [])) for r in result['results'])
            print(f"  Total issues found: {issues_count}")
            
            return True
        else:
            print(f"✗ Quality check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_validation_summary():
    """Test validation summary endpoint"""
    print("\n" + "=" * 60)
    print("Test 6: Validation Summary")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/validate/summary", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Got validation summary")
            print(f"  Total validations: {result['total_validations']}")
            print(f"  Drift detections: {result['recent_drift_detections']}")
            print(f"  Quality issues: {result['recent_quality_issues']}")
            print(f"  Engines monitored: {result['engines_monitored']}")
            return True
        else:
            print(f"✗ Summary failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("C-MAPSS Validation API Test Suite")
    print("=" * 60)
    
    # Check if preprocessed data exists
    if not Path("examples/cmapss/reference_baseline.json").exists():
        print("\n✗ Preprocessed data not found!")
        print("  Run: python scripts/preprocess_cmapss.py")
        sys.exit(1)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    
    if not results[0][1]:
        print("\n✗ API is not running. Cannot continue tests.")
        print("  Start API: uvicorn app.main:app --reload")
        sys.exit(1)
    
    results.append(("Store Reference", test_store_reference()))
    results.append(("Batch Validation", test_batch_validation()))
    results.append(("Drift Detection", test_drift_detection()))
    results.append(("Quality Check", test_quality_check()))
    results.append(("Validation Summary", test_validation_summary()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    print(f"\n  Total: {passed_count}/{len(results)} tests passed")
    
    if passed_count == len(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

