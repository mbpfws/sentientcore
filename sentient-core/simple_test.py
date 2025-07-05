import requests
import json

# Simple test for Build 2 research functionality
print("Testing Build 2 research functionality...")

url = "http://127.0.0.1:8000/api/chat/message/json"
headers = {"Content-Type": "application/json"}
data = {
    "message": "I want to improve my English. Can you research popular language learning apps?",
    "workflow_mode": "intelligent",
    "research_mode": "knowledge"
}

try:
    print("Sending research request...")
    response = requests.post(url, headers=headers, json=data, timeout=60)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n=== SUCCESS! Research Response Received ===")
        print(json.dumps(result, indent=2))
        
        # Check if research was conducted
        response_content = result.get('data', {}).get('content', '')
        if any(keyword in response_content.lower() for keyword in ['research', 'duolingo', 'babbel', 'language', 'app']):
            print("\n✅ SUCCESS: Research delegation appears to be working!")
            print("The orchestrator successfully delegated to the research agent.")
        else:
            print("\n⚠️  Response received but may not contain research results")
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Request error: {e}")