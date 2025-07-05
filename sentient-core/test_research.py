import requests
import json
import time

# Test server connectivity first
print("Testing server connectivity...")
try:
    response = requests.get("http://localhost:8000/docs")
    print(f"Server status: {response.status_code}")
except Exception as e:
    print(f"Server connection error: {e}")
    exit(1)

# Test the Build 2 research functionality
print("\nTesting Build 2 research functionality...")
url = "http://localhost:8000/chat/message/json"
headers = {"Content-Type": "application/json"}
data = {
    "message": "I want a solution for my weak English. Can you research some popular language learning apps?",
    "workflow_mode": "intelligent",
    "research_mode": "knowledge"
}

try:
    print("Sending research request...")
    response = requests.post(url, headers=headers, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n=== RESEARCH RESPONSE ===")
        print(json.dumps(result, indent=2))
        
        # Check if research was conducted
        if 'response' in result and 'research' in result['response'].lower():
            print("\n✅ SUCCESS: Research delegation appears to be working!")
        else:
            print("\n⚠️  WARNING: Response doesn't indicate research was conducted")
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Request error: {e}")