import requests
import json

def test_memory_endpoint():
    """Test the memory store endpoint directly"""
    url = "http://localhost:8000/api/core-services/memory/store"
    
    payload = {
        "content": "This is a test memory entry for debugging",
        "layer": "knowledge_synthesis",
        "memory_type": "synthesis",
        "metadata": {
            "source": "test",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Making request to: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"Success! Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_memory_endpoint()