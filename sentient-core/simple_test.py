#!/usr/bin/env python3
"""
Simple Test for Basic Functionality
Tests core endpoints and basic conversation flow
"""

import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

async def test_basic_functionality():
    """Test basic server functionality"""
    print("🚀 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test 1: Health check
            print("\n1. Testing health endpoint...")
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check: {data.get('message')}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return
            
            # Test 2: Basic chat
            print("\n2. Testing basic chat...")
            chat_payload = {
                "message": "Hello, I need help with a project",
                "workflow_mode": "intelligent"
            }
            
            async with session.post(f"{API_BASE}/chat/message", json=chat_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Chat response received")
                    print(f"   Session ID: {data.get('session_id', 'N/A')[:8]}...")
                    print(f"   Content preview: {data.get('content', '')[:100]}...")
                    session_id = data.get('session_id')
                else:
                    print(f"❌ Chat failed: {response.status}")
                    return
            
            # Test 3: SSE endpoint
            print("\n3. Testing SSE endpoint...")
            try:
                async with session.get(f"{API_BASE}/sse/events") as response:
                    if response.status == 200:
                        print("✅ SSE endpoint accessible")
                    else:
                        print(f"❌ SSE endpoint failed: {response.status}")
            except Exception as e:
                print(f"❌ SSE test error: {e}")
            
            # Test 4: Context retrieval (if session exists)
            if session_id:
                print("\n4. Testing context retrieval...")
                try:
                    async with session.get(f"{API_BASE}/chat/context/{session_id}") as response:
                        if response.status == 200:
                            print("✅ Context retrieval successful")
                        else:
                            print(f"❌ Context retrieval failed: {response.status}")
                except Exception as e:
                    print(f"❌ Context test error: {e}")
            
            # Test 5: Research trigger
            print("\n5. Testing research trigger...")
            research_payload = {
                "message": "Please research modern web development frameworks and best practices",
                "workflow_mode": "intelligent",
                "session_id": session_id
            }
            
            async with session.post(f"{API_BASE}/chat/message", json=research_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Research request processed")
                    
                    # Check for confirmation request
                    if data.get('message_type') == 'confirmation':
                        print("   🔄 Confirmation required for research")
                    else:
                        print("   📝 Direct response received")
                else:
                    print(f"❌ Research request failed: {response.status}")
            
            print("\n" + "=" * 40)
            print("✅ Basic functionality tests completed!")
            print(f"🌐 Server running at: {BASE_URL}")
            print(f"📡 API base: {API_BASE}")
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")

async def main():
    await test_basic_functionality()

if __name__ == "__main__":
    asyncio.run(main())