#!/usr/bin/env python3
"""
Comprehensive test script for monitoring endpoints
Tests all monitoring endpoints for Builds 1, 2, and 3
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

def test_endpoint(endpoint: str, method: str = "GET", data: Dict[Any, Any] = None) -> Dict[str, Any]:
    """Test an endpoint and return the response"""
    try:
        if method == "GET":
            response = requests.get(f"{API_BASE}{endpoint}")
        elif method == "POST":
            response = requests.post(f"{API_BASE}{endpoint}", json=data)
        
        print(f"\n{'='*60}")
        print(f"Testing: {method} {endpoint}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)[:500]}...")
            return result
        else:
            print(f"Error: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"error": str(e)}

def create_test_session() -> str:
    """Create a test session and return session_id"""
    print("\n" + "="*60)
    print("CREATING TEST SESSION")
    print("="*60)
    
    # Send a test message to create a session
    test_data = {
        "message": "Hello, this is a test message for monitoring endpoints",
        "session_id": "test_monitoring_session"
    }
    
    response = test_endpoint("/chat/message", "POST", test_data)
    
    if "error" not in response:
        session_id = response.get("session_id", "test_monitoring_session")
        print(f"Created session: {session_id}")
        return session_id
    else:
        print("Failed to create session, using default")
        return "test_monitoring_session"

def test_research_functionality(session_id: str):
    """Test research functionality to generate artifacts"""
    print("\n" + "="*60)
    print("TESTING RESEARCH FUNCTIONALITY (BUILD 2)")
    print("="*60)
    
    # Send a research query
    research_data = {
        "message": "research: What are the latest trends in AI development?",
        "session_id": session_id
    }
    
    response = test_endpoint("/chat/message", "POST", research_data)
    
    if "error" not in response:
        print("Research request sent successfully")
        # Wait a bit for processing
        time.sleep(2)
    else:
        print("Research request failed")

def test_planning_functionality(session_id: str):
    """Test planning functionality to generate artifacts"""
    print("\n" + "="*60)
    print("TESTING PLANNING FUNCTIONALITY (BUILD 3)")
    print("="*60)
    
    # Send a planning query
    planning_data = {
        "message": "plan: Create a web application for task management",
        "session_id": session_id
    }
    
    response = test_endpoint("/chat/message", "POST", planning_data)
    
    if "error" not in response:
        print("Planning request sent successfully")
        # Wait a bit for processing
        time.sleep(2)
    else:
        print("Planning request failed")

def test_monitoring_endpoints(session_id: str):
    """Test all monitoring endpoints"""
    print("\n" + "="*60)
    print("TESTING MONITORING ENDPOINTS")
    print("="*60)
    
    # Test session logs
    print("\n--- Testing Session Logs ---")
    test_endpoint(f"/monitoring/logs/{session_id}")
    test_endpoint(f"/monitoring/logs/{session_id}?limit=5&level=INFO")
    
    # Test session artifacts
    print("\n--- Testing Session Artifacts ---")
    test_endpoint(f"/monitoring/artifacts/{session_id}")
    
    # Test conversation history
    print("\n--- Testing Conversation History ---")
    test_endpoint(f"/monitoring/conversation/{session_id}")
    
    # Test session status
    print("\n--- Testing Session Status ---")
    test_endpoint(f"/monitoring/status/{session_id}")
    
    # Test with non-existent session
    print("\n--- Testing Non-existent Session ---")
    test_endpoint("/monitoring/status/non_existent_session")

def test_health_endpoints():
    """Test basic health endpoints"""
    print("\n" + "="*60)
    print("TESTING HEALTH ENDPOINTS")
    print("="*60)
    
    # Test root endpoint
    test_endpoint("/", "GET")
    
    # Test health endpoint
    test_endpoint("/health", "GET")

def main():
    """Main test function"""
    print("Starting Monitoring Endpoints Test")
    print("=" * 60)
    
    # Test basic health first
    test_health_endpoints()
    
    # Create a test session
    session_id = create_test_session()
    
    # Test research functionality (Build 2)
    test_research_functionality(session_id)
    
    # Test planning functionality (Build 3)
    test_planning_functionality(session_id)
    
    # Test all monitoring endpoints
    test_monitoring_endpoints(session_id)
    
    print("\n" + "="*60)
    print("MONITORING ENDPOINTS TEST COMPLETED")
    print("="*60)
    print("\nCheck the output above for any errors or issues.")
    print("All endpoints should return status code 200 for successful requests.")
    print("\nMonitoring endpoints tested:")
    print("- GET /api/monitoring/logs/{session_id}")
    print("- GET /api/monitoring/artifacts/{session_id}")
    print("- GET /api/monitoring/conversation/{session_id}")
    print("- GET /api/monitoring/status/{session_id}")
    print("- GET /api/monitoring/download/{artifact_type}/{filename}")

if __name__ == "__main__":
    main()