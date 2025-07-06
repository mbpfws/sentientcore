#!/usr/bin/env python3
"""
Debug Planning Transition - Focused Test

This script tests the specific planning transition logic
to identify why Layer 2 planning is not being triggered.
"""

import requests
import json
import time
import uuid
from pathlib import Path


def send_message(message: str, session_id: str, base_url: str = "http://localhost:8000"):
    """Send a message and return the response."""
    try:
        response = requests.post(
            f"{base_url}/api/chat/message/json",
            json={
                "message": message,
                "session_id": session_id
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "content": data.get("data", {}).get("content", ""),
                "message_type": data.get("data", {}).get("message_type", "text"),
                "metadata": data.get("data", {}).get("metadata", {})
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def check_layer_directories():
    """Check Layer 1 and Layer 2 directories."""
    layer1_path = Path("./memory/layer1_research_docs")
    layer2_path = Path("./memory/layer2_planning_docs")
    
    print(f"\n📁 Layer Status:")
    print(f"   Layer 1: {layer1_path.exists()} - {len(list(layer1_path.glob('*.md'))) if layer1_path.exists() else 0} files")
    print(f"   Layer 2: {layer2_path.exists()} - {len(list(layer2_path.glob('*'))) if layer2_path.exists() else 0} files")
    
    return layer1_path.exists() and len(list(layer1_path.glob('*.md'))) > 0


def main():
    """Debug the planning transition."""
    session_id = f"debug-planning-{uuid.uuid4().hex[:8]}"
    print(f"🔍 Debug Planning Transition")
    print(f"Session ID: {session_id}")
    print("=" * 50)
    
    # Check if we already have research files
    has_research = check_layer_directories()
    
    if not has_research:
        print("\n⚠️  No research files found. Creating research first...")
        research_msg = "Research language learning apps for developers. Focus on technical documentation challenges."
        print(f"Sending: {research_msg}")
        
        research_response = send_message(research_msg, session_id)
        if research_response["success"]:
            print("✅ Research request sent")
            print("⏳ Waiting 15 seconds for research...")
            time.sleep(15)
        else:
            print(f"❌ Research failed: {research_response['error']}")
            return
    
    # Check research status again
    has_research = check_layer_directories()
    if not has_research:
        print("❌ Still no research files. Cannot test planning transition.")
        return
    
    print(f"\n✅ Research files exist. Testing planning transition...")
    
    # Test different planning trigger messages
    planning_messages = [
        "Now create a PRD based on the research.",
        "I want to proceed with planning phase.",
        "Please create a Product Requirements Document.",
        "Let's move to the planning phase and create detailed requirements.",
        "Based on the research, can you create a comprehensive plan?"
    ]
    
    for i, planning_msg in enumerate(planning_messages, 1):
        print(f"\n🎯 Test {i}: {planning_msg}")
        
        response = send_message(planning_msg, session_id)
        if response["success"]:
            content_lower = response["content"].lower()
            
            # Check for planning indicators
            planning_indicators = ["prd", "requirements", "planning", "architecture", "document"]
            has_planning_content = any(indicator in content_lower for indicator in planning_indicators)
            
            print(f"   Response length: {len(response['content'])}")
            print(f"   Contains planning content: {has_planning_content}")
            
            if "layer 2" in content_lower or "planning phase" in content_lower:
                print("   🎉 Planning transition detected!")
            else:
                print("   ⚠️  No planning transition detected")
        else:
            print(f"   ❌ Request failed: {response['error']}")
        
        # Check if Layer 2 was created
        time.sleep(5)
        layer2_path = Path("./memory/layer2_planning_docs")
        if layer2_path.exists() and len(list(layer2_path.glob('*'))) > 0:
            print(f"   ✅ Layer 2 created with {len(list(layer2_path.glob('*')))} files!")
            return True
        else:
            print("   ❌ Layer 2 still not created")
    
    print("\n❌ All planning trigger tests failed")
    return False


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*50}")
    if success:
        print("✅ Planning transition working!")
    else:
        print("❌ Planning transition needs debugging")
        print("🔧 Check orchestrator logs for transition logic")