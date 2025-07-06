#!/usr/bin/env python3
"""
Test script to verify the created_at attribute fix for Message objects.
"""

import asyncio
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

async def test_created_at_fix():
    """Test that the created_at attribute error is resolved."""
    print("🔧 Testing created_at attribute fix...")
    print("=" * 50)
    
    try:
        # Test 1: Send a simple message to trigger the planning transition
        print("\n1. Testing message sending with planning transition...")
        
        response = requests.post(
            f"{BASE_URL}/chat/message/json",
            json={
                "message": "I want to build a simple Python calculator app",
                "session_id": "test-created-at-fix",
                "workflow_mode": "intelligent"
            },
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Message sent successfully")
            print(f"Response data keys: {list(data.keys())}")
            
            # Check if there's an error in the response
            if 'error' in data:
                print(f"❌ Error in response: {data['error']}")
                return False
            else:
                print(f"✅ No error in response")
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        # Test 2: Get chat history to verify created_at handling
        print("\n2. Testing chat history retrieval...")
        
        history_response = requests.get(
            f"{BASE_URL}/chat/history/test-created-at-fix",
            timeout=10
        )
        
        print(f"History response status: {history_response.status_code}")
        
        if history_response.status_code == 200:
            history_data = history_response.json()
            print(f"✅ Chat history retrieved successfully")
            
            if 'data' in history_data and 'messages' in history_data['data']:
                messages = history_data['data']['messages']
                print(f"Found {len(messages)} messages in history")
                
                # Check if messages have created_at field
                for i, msg in enumerate(messages):
                    if 'created_at' in msg:
                        print(f"✅ Message {i+1} has created_at: {msg['created_at']}")
                    else:
                        print(f"❌ Message {i+1} missing created_at field")
                        
            return True
        else:
            print(f"❌ History request failed with status {history_response.status_code}")
            print(f"Response: {history_response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_created_at_fix())
    if result:
        print("\n🎉 created_at fix test completed successfully!")
    else:
        print("\n💥 created_at fix test failed!")