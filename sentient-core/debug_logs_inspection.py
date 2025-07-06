#!/usr/bin/env python3
"""
Debug script to inspect the actual log sources and messages
to understand why planning transition is not working.
"""

import requests
import json
import time
import os

def send_message(message: str, session_id: str = "debug_session") -> dict:
    """Send a message to the backend API."""
    try:
        response = requests.post(
            "http://localhost:8000/api/chat/message/json",
            json={
                "message": message,
                "session_id": session_id
            },
            timeout=30
        )
        
        print(f"📡 API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Response Data Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            print(f"📊 Full Response Data: {data}")
            return data
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {}

def check_layer1_files():
    """Check what files exist in Layer 1 memory"""
    layer1_path = os.path.join(os.getcwd(), "memory", "layer1_research_docs")
    if os.path.exists(layer1_path):
        files = [f for f in os.listdir(layer1_path) if f.endswith('.md')]
        print(f"📁 Layer 1 files found: {files}")
        return len(files) > 0
    else:
        print("❌ Layer 1 directory does not exist")
        return False

def main():
    print("🔍 Debug: Inspecting logs and planning transition")
    print("=" * 60)
    
    # First, ensure we have research files
    if not check_layer1_files():
        print("\n🔬 Creating research request first...")
        research_response = send_message(
            "I want to research language learning apps. Can you help me understand the market, user needs, and technical requirements?"
        )
        print(f"Research response length: {len(research_response.get('response', ''))}")
        time.sleep(2)
        
        # Check again
        if not check_layer1_files():
            print("❌ Still no research files after research request")
            return
    
    print("\n🎯 Now testing planning transition with detailed log inspection...")
    
    # Send planning request
    planning_response = send_message(
        "Based on the research, can you create a detailed PRD and move to the planning phase?"
    )
    
    print(f"\n📊 Planning Response Analysis:")
    print(f"   Response length: {len(planning_response.get('response', ''))}")
    print(f"   Response preview: {planning_response.get('response', '')[:200]}...")
    
    # Check if logs are available in the response
    if 'logs' in planning_response:
        print(f"\n📋 Logs found in response:")
        for i, log in enumerate(planning_response['logs'][-10:]):
            print(f"   {i+1}. Source: '{log.get('source', 'unknown')}' - Message: '{log.get('message', '')}[:100]'")
    else:
        print("\n⚠️  No logs found in response")
    
    # Check if state information is available
    if 'state' in planning_response:
        state = planning_response['state']
        print(f"\n🔄 State Information:")
        print(f"   Planning state: {state.get('planning_state', 'unknown')}")
        print(f"   Next action: {state.get('next_action', 'unknown')}")
        print(f"   Messages count: {len(state.get('messages', []))}")
        print(f"   Logs count: {len(state.get('logs', []))}")
    
    # Check Layer 2 directory
    layer2_path = os.path.join(os.getcwd(), "memory", "layer2_planning_docs")
    if os.path.exists(layer2_path):
        layer2_files = [f for f in os.listdir(layer2_path) if f.endswith('.md')]
        print(f"\n📁 Layer 2 files: {layer2_files}")
        if layer2_files:
            print("✅ Planning transition successful!")
        else:
            print("❌ Layer 2 directory exists but no planning files created")
    else:
        print("\n❌ Layer 2 directory does not exist - planning transition failed")

if __name__ == "__main__":
    main()