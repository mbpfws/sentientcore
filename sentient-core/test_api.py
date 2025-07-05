import requests
import json
import uuid

def test_chat_api():
    url = "http://localhost:8000/api/chat/message/json"
    # Use a fresh session ID each time
    session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    payload = {
        "message": "I am a software developer and I have trouble with writing clear documentation in English. Can you research some solutions for me?",
        "session_id": session_id,
        "research_mode": "deep"
    }
    
    print(f"Using session ID: {session_id}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ API call successful!")
        else:
            print("❌ API call failed")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_chat_api()