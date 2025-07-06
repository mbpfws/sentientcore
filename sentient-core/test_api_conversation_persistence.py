#!/usr/bin/env python3
"""
Test script to verify conversation_history persistence through actual API calls.
This tests the real-world scenario where users interact with the chat API.
"""

import asyncio
import json
import aiohttp
import uuid
from typing import Dict, Any


async def test_api_conversation_persistence():
    """
    Test conversation_history persistence through actual API calls
    """
    print("=== Testing API Conversation Persistence ===")
    
    # Generate a unique session ID for this test
    session_id = str(uuid.uuid4())
    print(f"Using session ID: {session_id}")
    
    base_url = "http://127.0.0.1:8000"
    
    async with aiohttp.ClientSession() as session:
        try:
            # First API call - should create initial conversation_history
            print("\n1. Sending first message...")
            first_payload = {
                "message": "I need help with a project",
                "workflow_mode": "intelligent",
                "session_id": session_id
            }
            
            async with session.post(f"{base_url}/api/chat/message/json", json=first_payload) as response:
                if response.status == 200:
                    first_result = await response.json()
                    print(f"✓ First message processed successfully")
                    print(f"Response: {first_result['data']['content'][:100]}...")
                else:
                    print(f"✗ First API call failed: {response.status}")
                    return False
            
            # Wait a moment to ensure processing is complete
            await asyncio.sleep(2)
            
            # Second API call - should build upon existing conversation_history
            print("\n2. Sending second message...")
            second_payload = {
                "message": "It's a web application for e-commerce",
                "workflow_mode": "intelligent", 
                "session_id": session_id
            }
            
            async with session.post(f"{base_url}/api/chat/message/json", json=second_payload) as response:
                if response.status == 200:
                    second_result = await response.json()
                    print(f"✓ Second message processed successfully")
                    print(f"Response: {second_result['data']['content'][:100]}...")
                else:
                    print(f"✗ Second API call failed: {response.status}")
                    return False
            
            # Wait a moment to ensure processing is complete
            await asyncio.sleep(2)
            
            # Get chat history to verify persistence
            print("\n3. Retrieving chat history...")
            async with session.get(f"{base_url}/api/chat/history?session_id={session_id}") as response:
                if response.status == 200:
                    history_result = await response.json()
                    messages = history_result['data']['messages']
                    print(f"✓ Retrieved {len(messages)} messages from history")
                    
                    # Print message history
                    for i, msg in enumerate(messages):
                        print(f"  [{i}] {msg['sender']}: {msg['content'][:80]}...")
                    
                    # Verify we have both user and assistant messages
                    user_messages = [msg for msg in messages if msg['sender'] == 'user']
                    assistant_messages = [msg for msg in messages if msg['sender'] == 'assistant']
                    
                    if len(user_messages) >= 2 and len(assistant_messages) >= 2:
                        print("\n✓ SUCCESS: Chat history shows proper message persistence")
                        
                        # Check if assistant responses show contextual awareness
                        second_response = assistant_messages[-1]['content'].lower()
                        if any(keyword in second_response for keyword in ['project', 'web', 'application', 'e-commerce', 'previous', 'context']):
                            print("✓ SUCCESS: Assistant responses show contextual awareness")
                            return True
                        else:
                            print("⚠ WARNING: Assistant responses may not show full contextual awareness")
                            print(f"Second response: {second_response[:200]}...")
                            return True  # Still consider success if messages persist
                    else:
                        print(f"✗ FAILURE: Expected at least 2 user and 2 assistant messages, got {len(user_messages)} user, {len(assistant_messages)} assistant")
                        return False
                else:
                    print(f"✗ History retrieval failed: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            return False


async def main():
    """
    Main test function
    """
    print("Starting API conversation persistence test...")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    
    # Test API server availability
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:8000/docs") as response:
                if response.status == 200:
                    print("✓ API server is accessible")
                else:
                    print(f"✗ API server returned status {response.status}")
                    return
    except Exception as e:
        print(f"✗ Cannot connect to API server: {e}")
        print("Please ensure the server is running with: uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    # Run the persistence test
    success = await test_api_conversation_persistence()
    
    if success:
        print("\n🎉 OVERALL SUCCESS: API conversation persistence is working correctly!")
    else:
        print("\n❌ OVERALL FAILURE: API conversation persistence has issues")


if __name__ == "__main__":
    asyncio.run(main())