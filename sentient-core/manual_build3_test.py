#!/usr/bin/env python3
"""
Manual Build 3 Test - Interactive Testing

This script performs a manual test of Build 3 functionality
to verify the Architect Planner Agent and Layer 2 memory.
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


def check_memory_directories():
    """Check the status of memory directories."""
    layer1_path = Path("./memory/layer1_research_docs")
    layer2_path = Path("./memory/layer2_planning_docs")
    
    print("\n📁 Memory Directory Status:")
    print(f"   Layer 1 (Research): {layer1_path.exists()} - {len(list(layer1_path.glob('*.md'))) if layer1_path.exists() else 0} files")
    print(f"   Layer 2 (Planning): {layer2_path.exists()} - {len(list(layer2_path.glob('*.md'))) if layer2_path.exists() else 0} files")
    
    if layer1_path.exists():
        recent_files = [f for f in layer1_path.glob('*.md') if (time.time() - f.stat().st_mtime) < 3600]
        print(f"   Recent research files (last hour): {len(recent_files)}")
    
    if layer2_path.exists():
        recent_files = [f for f in layer2_path.glob('*.md') if (time.time() - f.stat().st_mtime) < 3600]
        print(f"   Recent planning files (last hour): {len(recent_files)}")


def main():
    """Run the manual Build 3 test."""
    session_id = f"manual-build3-{uuid.uuid4().hex[:8]}"
    print(f"🚀 Manual Build 3 Test")
    print(f"Session ID: {session_id}")
    print("=" * 60)
    
    # Step 1: Research Phase
    print("\n🔍 Step 1: Research Phase")
    research_query = (
        "I want to build a language learning app for software developers "
        "who struggle with technical documentation in English. "
        "Please research existing solutions and market analysis."
    )
    
    print(f"Sending: {research_query}")
    print("⏳ Waiting for research (may take 60-90 seconds)...")
    
    research_response = send_message(research_query, session_id)
    
    if research_response["success"]:
        print("✅ Research request sent successfully")
        print(f"Response length: {len(research_response['content'])} characters")
        print(f"Message type: {research_response['message_type']}")
        
        # Check for research completion indicators
        content_lower = research_response["content"].lower()
        research_indicators = ["research", "findings", "analysis", "solutions", "market"]
        has_research = any(indicator in content_lower for indicator in research_indicators)
        print(f"Contains research content: {has_research}")
        
        if "research completed" in content_lower or "report generated" in content_lower:
            print("🎉 Research phase completed!")
        else:
            print("⚠️  Research may still be in progress")
    else:
        print(f"❌ Research request failed: {research_response['error']}")
        return False
    
    # Wait a moment for processing
    print("\n⏳ Waiting 10 seconds for research processing...")
    time.sleep(10)
    
    # Check memory after research
    check_memory_directories()
    
    # Step 2: Planning Phase
    print("\n\n🎯 Step 2: Planning Phase")
    planning_query = (
        "Based on the research you just completed, can you create a detailed "
        "Product Requirements Document (PRD) for this language learning app? "
        "I want to proceed with the planning phase and get a comprehensive plan."
    )
    
    print(f"Sending: {planning_query}")
    print("⏳ Waiting for planning (may take 60-120 seconds)...")
    
    planning_response = send_message(planning_query, session_id)
    
    if planning_response["success"]:
        print("✅ Planning request sent successfully")
        print(f"Response length: {len(planning_response['content'])} characters")
        print(f"Message type: {planning_response['message_type']}")
        
        # Check for planning completion indicators
        content_lower = planning_response["content"].lower()
        planning_indicators = ["prd", "requirements", "planning", "architecture", "document"]
        has_planning = any(indicator in content_lower for indicator in planning_indicators)
        print(f"Contains planning content: {has_planning}")
        
        if "prd" in content_lower or "requirements document" in content_lower:
            print("🎉 Planning phase completed!")
        else:
            print("⚠️  Planning may still be in progress")
    else:
        print(f"❌ Planning request failed: {planning_response['error']}")
        return False
    
    # Wait for file system operations
    print("\n⏳ Waiting 15 seconds for file system operations...")
    time.sleep(15)
    
    # Final memory check
    print("\n\n📊 Final Results:")
    check_memory_directories()
    
    # Check for specific Build 3 artifacts
    layer2_path = Path("./memory/layer2_planning_docs")
    if layer2_path.exists():
        prd_files = list(layer2_path.glob("prd_*.md"))
        json_files = list(layer2_path.glob("prd_*.json"))
        
        print(f"\n🎯 Build 3 Artifacts:")
        print(f"   PRD Markdown files: {len(prd_files)}")
        print(f"   PRD JSON files: {len(json_files)}")
        
        if prd_files:
            print("\n📄 PRD Files:")
            for prd_file in prd_files[:3]:  # Show first 3
                print(f"   - {prd_file.name} ({prd_file.stat().st_size} bytes)")
        
        if prd_files or json_files:
            print("\n🎉 Build 3 SUCCESS: Architect Planner Agent created PRD artifacts!")
            return True
        else:
            print("\n⚠️  Build 3 PARTIAL: Directory exists but no PRD files found")
            return False
    else:
        print("\n❌ Build 3 FAILED: No Layer 2 planning directory created")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    if success:
        print("✅ Manual Build 3 test completed successfully!")
        print("🏗️  Architect Planner Agent is working correctly")
        print("💾 Layer 2 memory persistence is functional")
    else:
        print("❌ Manual Build 3 test failed")
        print("🔧 Build 3 may need debugging")
    
    print("\n🌐 Check the frontend for session details and artifacts")