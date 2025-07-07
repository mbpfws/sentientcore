#!/usr/bin/env python3
"""
Test script for simple HTTP server connectivity
"""

import requests
import json
import time

def test_health():
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        print(f"Health check status: {response.status_code}")
        print(f"Health check response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_chat():
    try:
        payload = {
            "message": "Hello, this is a test message",
            "session_id": "test_session"
        }
        response = requests.post(
            "http://localhost:8002/api/chat/message/json",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"Chat test status: {response.status_code}")
        print(f"Chat test response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Chat test failed: {e}")
        return False

def test_root():
    try:
        response = requests.get("http://localhost:8002/", timeout=5)
        print(f"Root endpoint status: {response.status_code}")
        print(f"Root endpoint response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing simple HTTP server connectivity on port 8002...")
    print("Waiting 3 seconds for server to be ready...")
    time.sleep(3)
    
    root_ok = test_root()
    health_ok = test_health()
    chat_ok = test_chat()
    
    print("\n=== Test Results ===")
    print(f"Root endpoint: {'✅' if root_ok else '❌'}")
    print(f"Health endpoint: {'✅' if health_ok else '❌'}")
    print(f"Chat endpoint: {'✅' if chat_ok else '❌'}")
    
    if root_ok and health_ok and chat_ok:
        print("\n✅ All tests passed! Simple HTTP server is working.")
    else:
        print("\n❌ Some tests failed.")
        if not any([root_ok, health_ok, chat_ok]):
            print("💡 Server might not be running or accessible on port 8002")