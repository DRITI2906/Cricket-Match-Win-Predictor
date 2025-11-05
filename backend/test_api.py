"""
Simple test script to verify the API is working
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_prediction():
    """Test the prediction endpoint"""
    print("Testing prediction endpoint...")
    
    match_data = {
        "team1": "India",
        "team2": "Australia",
        "venue": "Melbourne Cricket Ground",
        "toss_winner": "India",
        "toss_decision": "bat",
        "match_type": "ODI",
        "runs_required": 150,
        "balls_remaining": 120,
        "wickets_in_hand": 8,
        "target_match": 250,
        "current_run_rate": 6.0,
        "required_run_rate": 7.5
    }
    
    response = requests.post(f"{BASE_URL}/api/predict", json=match_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction Result:")
        print(f"  Winner: {result['winner']}")
        print(f"  Probability: {result['probability']:.2%}")
        print(f"  Confidence: {result['confidence']}")
        print(f"\n  Top SHAP Features:")
        for shap in result['shap_explanation'][:5]:
            print(f"    - {shap['feature']}: {shap['value']:.3f} ({shap['impact']})")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    print("=" * 60)
    print("Cricket Prediction API Test")
    print("=" * 60)
    print()
    
    try:
        # Test health check
        if test_health_check():
            print("✓ Health check passed\n")
        else:
            print("✗ Health check failed\n")
            return
        
        # Test prediction
        if test_prediction():
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Prediction test failed")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API")
        print("Make sure the server is running on http://localhost:8000")
        print("Run: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
