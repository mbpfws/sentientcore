#!/usr/bin/env python3

import requests
import json

def test_basic_chat():
    """Test basic chat functionality"""
    url = "http://localhost:8000/api/chat/message/json"
    
    # Simple chat message
    payload = {
        "message": "Hello, can you help me?",
        "conversation_id": "test-conversation-001"
    }
    
    try:
        print("Testing basic chat functionality...")
        print(f"Sending request to: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS: Chat response received")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"\n❌ FAILED: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_chat()
    if success:
        print("\n🎉 Basic chat is working!")
    else:
        print("\n💥 Basic chat is NOT working!")