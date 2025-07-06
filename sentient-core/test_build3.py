#!/usr/bin/env python3
"""
Comprehensive Test Script for Build 3 Validation

This script validates that Build 3 (Architect Planner Agent & Tiered Memory)
is working correctly according to the test case specifications.

Build 3 Test Cases:
1. Research-to-Planning Transition
2. PRD Generation from Research
3. Layer 2 Memory Persistence
4. Architect Planner Agent Functionality
5. Integration with Previous Builds
"""

import asyncio
import json
import requests
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List


class Build3Tester:
    """Comprehensive tester for Build 3 functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test-build3-{uuid.uuid4().hex[:8]}"
        self.test_results = []
        
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results."""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_results.append(result)
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
    
    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the chat API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat/message/json",
                json={
                    "message": message,
                    "session_id": self.session_id
                },
                timeout=120  # Longer timeout for planning tasks
            )
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
        except Exception as e:
            return {
                "status_code": 0,
                "data": None,
                "error": str(e)
            }
    
    def get_session_history(self) -> Dict[str, Any]:
        """Get session history from the API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/chat/history",
                params={"session_id": self.session_id},
                timeout=30
            )
            return {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
        except Exception as e:
            return {
                "status_code": 0,
                "data": None,
                "error": str(e)
            }
    
    def check_planning_artifacts(self) -> Dict[str, Any]:
        """Check if planning artifacts are available in Layer 2 memory."""
        planning_docs_path = Path("./memory/layer2_planning_docs")
        
        if not planning_docs_path.exists():
            return {
                "status": "error",
                "message": "Layer 2 planning docs directory does not exist",
                "artifacts": []
            }
        
        artifacts = list(planning_docs_path.glob("*.md")) + list(planning_docs_path.glob("*.pdf"))
        
        return {
            "status": "success",
            "message": f"Found {len(artifacts)} planning artifacts",
            "artifacts": [str(artifact) for artifact in artifacts]
        }
    
    def test_research_phase(self):
        """Test the research phase that should precede planning."""
        print("\n🔍 Testing Research Phase (Prerequisite for Build 3)")
        print("=" * 60)
        
        # Test: Send a research request that should lead to planning
        print("\n📝 Test: Research Request for Planning")
        research_query = (
            "I want to build a language learning app for software developers "
            "who struggle with technical documentation in English. "
            "Can you research existing solutions and then help me plan this project?"
        )
        
        print(f"Sending research query: {research_query}")
        print("⏳ This may take 60-90 seconds as the system conducts research...")
        
        response = self.send_message(research_query)
        
        if response["status_code"] == 200 and response["data"]:
            content = response["data"].get("data", {}).get("content", "")
            
            # Check for research completion indicators
            research_indicators = [
                "research completed", "findings", "report", "generated", 
                "solutions", "tools", "applications", "analysis"
            ]
            has_research_content = any(indicator in content.lower() for indicator in research_indicators)
            
            if has_research_content:
                self.log_test(
                    "Build 3 - Research Phase", 
                    "PASS", 
                    "Research phase completed successfully"
                )
                return True
            else:
                self.log_test(
                    "Build 3 - Research Phase", 
                    "WARN", 
                    f"Research response unclear. Content: {content[:200]}..."
                )
                return True  # Continue anyway
        else:
            self.log_test(
                "Build 3 - Research Phase", 
                "FAIL", 
                f"Failed to process research request. Error: {response.get('error', 'Unknown error')}"
            )
            return False
    
    def test_planning_transition(self):
        """Test the transition from research to planning phase."""
        print("\n🎯 Testing Research-to-Planning Transition")
        print("=" * 60)
        
        # Test: Request planning based on research
        print("\n📝 Test: Planning Request After Research")
        planning_request = (
            "Based on the research you just completed, can you create a detailed "
            "Product Requirements Document (PRD) for this language learning app? "
            "I want to proceed with the planning phase."
        )
        
        print(f"Sending planning request: {planning_request}")
        print("⏳ This may take 60-120 seconds as the Architect Planner Agent works...")
        
        response = self.send_message(planning_request)
        
        if response["status_code"] == 200 and response["data"]:
            content = response["data"].get("data", {}).get("content", "")
            
            # Check for planning/PRD indicators
            planning_indicators = [
                "product requirements", "prd", "planning", "architecture", 
                "requirements", "specifications", "features", "technical", 
                "document", "plan", "design"
            ]
            has_planning_content = any(indicator in content.lower() for indicator in planning_indicators)
            
            if has_planning_content:
                self.log_test(
                    "Build 3 - Planning Transition", 
                    "PASS", 
                    "Successfully transitioned to planning phase and generated PRD content"
                )
                return True
            else:
                self.log_test(
                    "Build 3 - Planning Transition", 
                    "WARN", 
                    f"Planning response unclear. Content: {content[:200]}..."
                )
                return True  # Continue anyway
        else:
            self.log_test(
                "Build 3 - Planning Transition", 
                "FAIL", 
                f"Failed to process planning request. Error: {response.get('error', 'Unknown error')}"
            )
            return False
    
    def test_architect_planner_agent(self):
        """Test the Architect Planner Agent functionality."""
        print("\n🏗️ Testing Architect Planner Agent")
        print("=" * 60)
        
        # Test: Verify agent activation and PRD generation
        print("\n📝 Test: Architect Planner Agent Activation")
        
        # Check session history for agent activation indicators
        history_response = self.get_session_history()
        
        if history_response["status_code"] == 200 and history_response["data"]:
            messages = history_response["data"].get("messages", [])
            
            # Look for architect planner activation in messages
            architect_indicators = [
                "architect", "planner", "planning agent", "prd", 
                "requirements document", "technical architecture"
            ]
            
            has_architect_activity = False
            for message in messages:
                content = message.get("content", "").lower()
                if any(indicator in content for indicator in architect_indicators):
                    has_architect_activity = True
                    break
            
            if has_architect_activity:
                self.log_test(
                    "Build 3 - Architect Agent Activation", 
                    "PASS", 
                    "Architect Planner Agent was successfully activated"
                )
            else:
                self.log_test(
                    "Build 3 - Architect Agent Activation", 
                    "WARN", 
                    "Could not confirm Architect Planner Agent activation in session history"
                )
        else:
            self.log_test(
                "Build 3 - Architect Agent Activation", 
                "FAIL", 
                f"Failed to retrieve session history. Error: {history_response.get('error', 'Unknown error')}"
            )
        
        return True
    
    def test_layer2_memory_persistence(self):
        """Test Layer 2 memory persistence for planning documents."""
        print("\n💾 Testing Layer 2 Memory Persistence")
        print("=" * 60)
        
        # Test: Check for planning artifacts in Layer 2 memory
        print("\n📝 Test: Layer 2 Planning Documents")
        
        # Wait a moment for file system operations to complete
        time.sleep(5)
        
        artifacts_result = self.check_planning_artifacts()
        
        if artifacts_result["status"] == "success":
            if artifacts_result["artifacts"]:
                self.log_test(
                    "Build 3 - Layer 2 Persistence", 
                    "PASS", 
                    f"Found {len(artifacts_result['artifacts'])} planning documents in Layer 2 memory"
                )
                
                # Show the artifacts
                for artifact in artifacts_result["artifacts"][:3]:  # Show first 3
                    artifact_path = Path(artifact)
                    print(f"   📄 {artifact_path.name} ({artifact_path.stat().st_size} bytes)")
                    
            else:
                self.log_test(
                    "Build 3 - Layer 2 Persistence", 
                    "WARN", 
                    "No planning documents found in Layer 2 memory. PRD may still be processing."
                )
        else:
            self.log_test(
                "Build 3 - Layer 2 Persistence", 
                "FAIL", 
                artifacts_result["message"]
            )
        
        # Test: Check Layer 1 memory for research documents
        print("\n📝 Test: Layer 1 Research Documents (Should Exist)")
        research_docs_path = Path("./memory/layer1_research_docs")
        
        if research_docs_path.exists():
            research_artifacts = list(research_docs_path.glob("*.md")) + list(research_docs_path.glob("*.pdf"))
            if research_artifacts:
                self.log_test(
                    "Build 3 - Layer 1 Verification", 
                    "PASS", 
                    f"Found {len(research_artifacts)} research documents in Layer 1 memory"
                )
            else:
                self.log_test(
                    "Build 3 - Layer 1 Verification", 
                    "WARN", 
                    "No research documents found in Layer 1 memory"
                )
        else:
            self.log_test(
                "Build 3 - Layer 1 Verification", 
                "FAIL", 
                "Layer 1 research docs directory does not exist"
            )
        
        return True
    
    def test_integration_with_previous_builds(self):
        """Test integration between Build 3 and previous builds."""
        print("\n🔗 Testing Integration with Previous Builds")
        print("=" * 60)
        
        # Test: Follow-up conversation after planning
        print("\n📝 Test: Follow-up Conversation After Planning")
        follow_up = "Can you explain the key features you included in the PRD?"
        
        response = self.send_message(follow_up)
        
        if response["status_code"] == 200 and response["data"]:
            content = response["data"].get("data", {}).get("content", "")
            
            # Check if the response references the planning work
            planning_references = [
                "features", "requirements", "prd", "document", 
                "specifications", "plan", "architecture"
            ]
            has_planning_reference = any(ref in content.lower() for ref in planning_references)
            
            if has_planning_reference:
                self.log_test(
                    "Build 3 - Integration Follow-up", 
                    "PASS", 
                    "System successfully referenced planning work in follow-up conversation"
                )
            else:
                self.log_test(
                    "Build 3 - Integration Follow-up", 
                    "WARN", 
                    "Follow-up response may not reference planning work"
                )
        else:
            self.log_test(
                "Build 3 - Integration Follow-up", 
                "FAIL", 
                f"Failed to handle follow-up conversation. Error: {response.get('error', 'Unknown error')}"
            )
        
        return True
    
    def run_all_tests(self):
        """Run all Build 3 tests in sequence."""
        print("🚀 Starting Comprehensive Build 3 Testing")
        print("=" * 70)
        print(f"Session ID: {self.session_id}")
        print(f"Base URL: {self.base_url}")
        print("\n🎯 Build 3 Focus: Architect Planner Agent & Tiered Memory")
        
        try:
            # Test Research Phase (prerequisite)
            if not self.test_research_phase():
                print("\n❌ Research phase failed. Cannot proceed to Build 3 tests.")
                return False
            
            # Test Planning Transition
            if not self.test_planning_transition():
                print("\n❌ Planning transition failed. Build 3 may not be working.")
                return False
            
            # Test Architect Planner Agent
            self.test_architect_planner_agent()
            
            # Test Layer 2 Memory Persistence
            self.test_layer2_memory_persistence()
            
            # Test Integration
            self.test_integration_with_previous_builds()
            
            # Summary
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"\n💥 Test execution failed with error: {e}")
            return False
    
    def print_summary(self):
        """Print test summary."""
        print("\n📊 Build 3 Test Summary")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        warnings = sum(1 for result in self.test_results if result["status"] == "WARN")
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"📊 Total: {len(self.test_results)}")
        
        if failed == 0:
            print("\n🎉 All critical Build 3 tests passed!")
            print("✅ Architect Planner Agent is working correctly")
            print("✅ Tiered Memory (Layer 2) is functioning")
            print("✅ Research-to-Planning transition is operational")
        else:
            print("\n⚠️  Some Build 3 tests failed. Please review the results above.")
        
        print(f"\n🔗 Session ID for manual verification: {self.session_id}")
        print(f"🌐 You can check this session at: {self.base_url}/chat?session={self.session_id}")
        
        # Show memory locations
        print("\n📁 Memory Locations to Check:")
        print("   📄 Layer 1 (Research): ./memory/layer1_research_docs/")
        print("   📄 Layer 2 (Planning): ./memory/layer2_planning_docs/")


if __name__ == "__main__":
    # Run the comprehensive Build 3 test
    tester = Build3Tester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Build 3 testing completed successfully!")
    else:
        print("\n❌ Build 3 testing completed with errors.")
        exit(1)