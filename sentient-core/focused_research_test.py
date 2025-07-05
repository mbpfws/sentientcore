#!/usr/bin/env python3
"""
Focused Research Test - Specifically test research agent functionality
"""

import requests
import json
import time
import uuid
from pathlib import Path

def test_research_functionality():
    """Test research functionality with detailed monitoring."""
    base_url = "http://localhost:8000"
    session_id = f"research-test-{uuid.uuid4().hex[:8]}"
    
    print(f"🔬 Testing Research Agent Functionality")
    print(f"Session ID: {session_id}")
    print("=" * 60)
    
    # Clear research directory first
    research_dir = Path("./memory/layer1_research_docs")
    research_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Research directory: {research_dir.absolute()}")
    print(f"📊 Files before test: {len(list(research_dir.glob('*')))}")
    
    # Send a clear research request
    research_query = "Please conduct research on the best practices for API documentation in software development. I need comprehensive information about tools, techniques, and industry standards."
    
    print(f"\n📝 Sending research query:")
    print(f"   {research_query}")
    print("\n⏳ Waiting for response (this may take 30-90 seconds)...")
    
    try:
        response = requests.post(
            f"{base_url}/api/chat/message/json",
            json={
                "message": research_query,
                "session_id": session_id
            },
            timeout=120  # Increased timeout for research
        )
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("data", {}).get("content", "")
            
            print(f"\n📄 Response Content (first 500 chars):")
            print(f"   {content[:500]}...")
            
            # Check for research indicators
            research_keywords = [
                "research", "findings", "report", "documentation", 
                "tools", "practices", "standards", "generated", "saved"
            ]
            
            found_keywords = [kw for kw in research_keywords if kw.lower() in content.lower()]
            print(f"\n🔍 Research keywords found: {found_keywords}")
            
            # Wait a bit for file system operations
            print("\n⏳ Waiting 5 seconds for file operations...")
            time.sleep(5)
            
            # Check artifacts
            print(f"\n📊 Files after test: {len(list(research_dir.glob('*')))}")
            
            artifacts = list(research_dir.glob("*.md")) + list(research_dir.glob("*.pdf"))
            if artifacts:
                print(f"\n✅ Found {len(artifacts)} research artifacts:")
                for artifact in artifacts:
                    size = artifact.stat().st_size
                    print(f"   📄 {artifact.name} ({size} bytes)")
                    
                    # Show first few lines of the artifact
                    if artifact.suffix == '.md' and size > 0:
                        try:
                            with open(artifact, 'r', encoding='utf-8') as f:
                                first_lines = f.read(200)
                                print(f"      Preview: {first_lines}...")
                        except Exception as e:
                            print(f"      Error reading file: {e}")
            else:
                print("\n⚠️  No research artifacts found in file system")
            
            # Test API artifacts endpoint
            print("\n🌐 Testing artifacts API endpoint...")
            artifacts_response = requests.get(f"{base_url}/api/research/artifacts")
            print(f"   Status: {artifacts_response.status_code}")
            
            if artifacts_response.status_code == 200:
                api_artifacts = artifacts_response.json().get("data", {}).get("artifacts", [])
                print(f"   API artifacts count: {len(api_artifacts)}")
                for artifact in api_artifacts[:3]:
                    print(f"   📄 {artifact.get('filename', 'Unknown')}")
            
            return True
            
        else:
            print(f"\n❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n💥 Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_research_functionality()
    if success:
        print("\n✅ Research test completed")
    else:
        print("\n❌ Research test failed")
        exit(1)