#!/usr/bin/env python3

import asyncio
import requests
import json

async def test_memory_store():
    """Test the memory store endpoint"""
    
    # Test data
    test_data = {
        "layer": "knowledge_synthesis",
        "memory_type": "documentation", 
        "content": "Test memory content for debugging",
        "metadata": {
            "source": "test_script",
            "timestamp": "2024-01-01T00:00:00Z"
        },
        "tags": ["test", "memory", "debug"]
    }
    
    try:
        # Test memory store
        print("Testing memory store endpoint...")
        response = requests.post(
            "http://localhost:8000/api/core-services/memory/store",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Memory store test passed!")
            return True
        else:
            print("❌ Memory store test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing memory store: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_memory_store())