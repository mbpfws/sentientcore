#!/usr/bin/env python3
"""
End-to-End Test for Sentient Core Application
Tests the complete user workflow from frontend to backend integration
"""

import requests
import json
import time
from typing import Dict, Any

def test_backend_health():
    """Test if backend server is healthy and responding"""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=10)
        print(f"✓ Backend health check: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Backend health check failed: {e}")
        return False

def test_frontend_health():
    """Test if frontend server is healthy and responding"""
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        print(f"✓ Frontend health check: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Frontend health check failed: {e}")
        return False

def test_chat_api_endpoint():
    """Test the chat API endpoint with different research modes"""
    chat_url = "http://localhost:8000/api/chat"
    
    test_cases = [
        {
            "name": "Knowledge Research",
            "payload": {
                "message": "What is artificial intelligence?",
                "research_mode": "knowledge"
            }
        },
        {
            "name": "Deep Research", 
            "payload": {
                "message": "Explain machine learning algorithms in detail",
                "research_mode": "deep"
            }
        },
        {
            "name": "Best-in-class Research",
            "payload": {
                "message": "What are the latest developments in AI?",
                "research_mode": "best-in-class"
            }
        },
        {
            "name": "Regular Chat",
            "payload": {
                "message": "Hello, how are you?"
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        try:
            print(f"\nTesting {test_case['name']}...")
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = requests.post(
                chat_url, 
                json=test_case["payload"], 
                headers=headers,
                timeout=30
            )
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    print(f"  Response Type: {type(response_data)}")
                    
                    # Check if response contains expected fields
                    if isinstance(response_data, dict):
                        if "response" in response_data:
                            print(f"  Response Length: {len(response_data['response'])} chars")
                            print(f"  ✓ {test_case['name']} - SUCCESS")
                            results.append({"test": test_case['name'], "status": "PASS", "details": "Valid response received"})
                        else:
                            print(f"  ✗ {test_case['name']} - Missing 'response' field")
                            results.append({"test": test_case['name'], "status": "FAIL", "details": "Missing response field"})
                    else:
                        print(f"  ✗ {test_case['name']} - Invalid response format")
                        results.append({"test": test_case['name'], "status": "FAIL", "details": "Invalid response format"})
                        
                except json.JSONDecodeError as e:
                    print(f"  ✗ {test_case['name']} - JSON decode error: {e}")
                    results.append({"test": test_case['name'], "status": "FAIL", "details": f"JSON decode error: {e}"})
            else:
                print(f"  ✗ {test_case['name']} - HTTP {response.status_code}")
                print(f"  Response: {response.text[:200]}...")
                results.append({"test": test_case['name'], "status": "FAIL", "details": f"HTTP {response.status_code}"})
                
        except requests.exceptions.RequestException as e:
            print(f"  ✗ {test_case['name']} - Request failed: {e}")
            results.append({"test": test_case['name'], "status": "FAIL", "details": f"Request failed: {e}"})
        
        # Small delay between requests
        time.sleep(1)
    
    return results

def test_api_documentation():
    """Test if API documentation endpoints are accessible"""
    endpoints = [
        "/docs",
        "/openapi.json"
    ]
    
    results = []
    
    for endpoint in endpoints:
        try:
            url = f"http://localhost:8000{endpoint}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"✓ API Documentation {endpoint}: Accessible")
                results.append({"endpoint": endpoint, "status": "PASS"})
            else:
                print(f"✗ API Documentation {endpoint}: HTTP {response.status_code}")
                results.append({"endpoint": endpoint, "status": "FAIL"})
                
        except Exception as e:
            print(f"✗ API Documentation {endpoint}: {e}")
            results.append({"endpoint": endpoint, "status": "FAIL"})
    
    return results

def main():
    """Run comprehensive end-to-end tests"""
    print("🚀 Starting End-to-End Tests for Sentient Core Application")
    print("=" * 60)
    
    # Test server health
    print("\n1. Testing Server Health...")
    backend_healthy = test_backend_health()
    frontend_healthy = test_frontend_health()
    
    if not backend_healthy:
        print("❌ Backend server is not healthy. Aborting tests.")
        return
    
    if not frontend_healthy:
        print("❌ Frontend server is not healthy. Aborting tests.")
        return
    
    print("✅ Both servers are healthy!")
    
    # Test API documentation
    print("\n2. Testing API Documentation...")
    doc_results = test_api_documentation()
    
    # Test chat functionality
    print("\n3. Testing Chat API Functionality...")
    chat_results = test_chat_api_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    # Documentation results
    print("\nAPI Documentation:")
    for result in doc_results:
        status_icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"  {status_icon} {result['endpoint']}: {result['status']}")
    
    # Chat API results
    print("\nChat API Tests:")
    passed_tests = 0
    total_tests = len(chat_results)
    
    for result in chat_results:
        status_icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"  {status_icon} {result['test']}: {result['status']}")
        if result["status"] == "PASS":
            passed_tests += 1
        else:
            print(f"    Details: {result['details']}")
    
    print(f"\nOverall Chat API Success Rate: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL END-TO-END TESTS PASSED!")
        print("✅ The application is ready for user interaction!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        print("❌ Some functionality may not work as expected.")
    
    print("\n" + "=" * 60)
    print("End-to-end testing completed.")

if __name__ == "__main__":
    main()