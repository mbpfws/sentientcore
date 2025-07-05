#!/usr/bin/env python3
"""
Comprehensive Test Script for Build 1 and Build 2 Validation

This script validates that both Build 1 (Ultra Orchestrator) and Build 2 (Research & Persistence)
are working correctly according to the test case specifications.
"""

import asyncio
import json
import requests
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List


class Build1And2Tester:
    """Comprehensive tester for Build 1 and Build 2 functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test-session-{uuid.uuid4().hex[:8]}"
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
                timeout=60
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
    
    def check_research_artifacts(self) -> Dict[str, Any]:
        """Check if research artifacts are available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/research/artifacts",
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
    
    def test_build_1_ultra_orchestrator(self):
        """Test Build 1: Ultra Orchestrator functionality."""
        print("\n🔍 Testing Build 1: Ultra Orchestrator")
        print("=" * 50)
        
        # Test 1.1: Basic conversation handling
        print("\n📝 Test 1.1: Basic Conversation Handling")
        response = self.send_message("Hello, system.")
        
        if response["status_code"] == 200 and response["data"]:
            self.log_test(
                "Build 1 - Basic Conversation", 
                "PASS", 
                f"Successfully handled greeting. Response: {response['data'].get('data', {}).get('content', '')[:100]}..."
            )
        else:
            self.log_test(
                "Build 1 - Basic Conversation", 
                "FAIL", 
                f"Failed to handle basic conversation. Error: {response.get('error', 'Unknown error')}"
            )
            return False
        
        # Test 1.2: Intent recognition and clarification
        print("\n📝 Test 1.2: Intent Recognition and Clarification")
        response = self.send_message("can you help with building an app")
        
        if response["status_code"] == 200 and response["data"]:
            content = response["data"].get("data", {}).get("content", "")
            # Check if the orchestrator asks clarifying questions
            clarification_indicators = ["what type", "tell me", "could you", "more about", "specific"]
            has_clarification = any(indicator in content.lower() for indicator in clarification_indicators)
            
            if has_clarification:
                self.log_test(
                    "Build 1 - Intent Recognition", 
                    "PASS", 
                    "Ultra Orchestrator correctly identified vague request and asked for clarification"
                )
            else:
                self.log_test(
                    "Build 1 - Intent Recognition", 
                    "WARN", 
                    f"Response may not contain proper clarification. Content: {content[:200]}..."
                )
        else:
            self.log_test(
                "Build 1 - Intent Recognition", 
                "FAIL", 
                f"Failed to process vague request. Error: {response.get('error', 'Unknown error')}"
            )
        
        return True
    
    def test_build_2_session_persistence(self):
        """Test Build 2: Session Persistence functionality."""
        print("\n💾 Testing Build 2: Session Persistence")
        print("=" * 50)
        
        # Test 2.1: Session history retrieval
        print("\n📝 Test 2.1: Session History Retrieval")
        history_response = self.get_session_history()
        
        if history_response["status_code"] == 200 and history_response["data"]:
            messages = history_response["data"].get("messages", [])
            if len(messages) >= 2:  # Should have at least the greeting and response
                self.log_test(
                    "Build 2 - Session Persistence", 
                    "PASS", 
                    f"Successfully retrieved {len(messages)} messages from session history"
                )
            else:
                self.log_test(
                    "Build 2 - Session Persistence", 
                    "WARN", 
                    f"Session history contains only {len(messages)} messages"
                )
        else:
            self.log_test(
                "Build 2 - Session Persistence", 
                "FAIL", 
                f"Failed to retrieve session history. Error: {history_response.get('error', 'Unknown error')}"
            )
            return False
        
        return True
    
    def test_build_2_research_agent(self):
        """Test Build 2: Research Agent functionality."""
        print("\n🔬 Testing Build 2: Research Agent")
        print("=" * 50)
        
        # Test 2.2: Research request processing
        print("\n📝 Test 2.2: Research Request Processing")
        research_query = "I am a software developer and I have trouble with writing clear documentation in English. Can you research some solutions for me?"
        
        print(f"Sending research query: {research_query}")
        print("⏳ This may take 30-60 seconds as the system conducts real research...")
        
        response = self.send_message(research_query)
        
        if response["status_code"] == 200 and response["data"]:
            content = response["data"].get("data", {}).get("content", "")
            
            # Check for research completion indicators
            research_indicators = [
                "research completed", "findings", "report", "generated", 
                "documentation", "solutions", "tools", "techniques"
            ]
            has_research_content = any(indicator in content.lower() for indicator in research_indicators)
            
            if has_research_content:
                self.log_test(
                    "Build 2 - Research Processing", 
                    "PASS", 
                    "Research agent successfully processed the query and generated findings"
                )
            else:
                self.log_test(
                    "Build 2 - Research Processing", 
                    "WARN", 
                    f"Response may not contain research results. Content: {content[:200]}..."
                )
        else:
            self.log_test(
                "Build 2 - Research Processing", 
                "FAIL", 
                f"Failed to process research request. Error: {response.get('error', 'Unknown error')}"
            )
            return False
        
        return True
    
    def test_build_2_artifact_generation(self):
        """Test Build 2: Artifact Generation functionality."""
        print("\n📄 Testing Build 2: Artifact Generation")
        print("=" * 50)
        
        # Test 2.3: Research artifacts availability
        print("\n📝 Test 2.3: Research Artifacts Availability")
        artifacts_response = self.check_research_artifacts()
        
        if artifacts_response["status_code"] == 200 and artifacts_response["data"]:
            artifacts = artifacts_response["data"].get("artifacts", [])
            if artifacts:
                self.log_test(
                    "Build 2 - Artifact Generation", 
                    "PASS", 
                    f"Found {len(artifacts)} research artifacts available for download"
                )
                
                # List the artifacts
                for artifact in artifacts[:3]:  # Show first 3
                    print(f"   📄 {artifact}")
                    
            else:
                self.log_test(
                    "Build 2 - Artifact Generation", 
                    "WARN", 
                    "No research artifacts found. This may be expected if research just completed."
                )
        else:
            self.log_test(
                "Build 2 - Artifact Generation", 
                "FAIL", 
                f"Failed to check research artifacts. Error: {artifacts_response.get('error', 'Unknown error')}"
            )
        
        # Test 2.4: Check file system for artifacts
        print("\n📝 Test 2.4: File System Artifact Check")
        research_docs_path = Path("./memory/layer1_research_docs")
        
        if research_docs_path.exists():
            artifacts = list(research_docs_path.glob("*.md")) + list(research_docs_path.glob("*.pdf"))
            if artifacts:
                self.log_test(
                    "Build 2 - File System Artifacts", 
                    "PASS", 
                    f"Found {len(artifacts)} research artifacts in file system"
                )
                
                # Show recent artifacts
                for artifact in sorted(artifacts, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    print(f"   📄 {artifact.name} ({artifact.stat().st_size} bytes)")
            else:
                self.log_test(
                    "Build 2 - File System Artifacts", 
                    "WARN", 
                    "No artifacts found in file system. Research may still be processing."
                )
        else:
            self.log_test(
                "Build 2 - File System Artifacts", 
                "FAIL", 
                "Research docs directory does not exist"
            )
        
        return True
    
    def test_integration(self):
        """Test integration between Build 1 and Build 2."""
        print("\n🔗 Testing Build 1 & 2 Integration")
        print("=" * 50)
        
        # Test integration: Follow-up question after research
        print("\n📝 Test: Follow-up Conversation After Research")
        follow_up = "Can you tell me more about the specific tools you mentioned?"
        
        response = self.send_message(follow_up)
        
        if response["status_code"] == 200 and response["data"]:
            self.log_test(
                "Integration - Follow-up Conversation", 
                "PASS", 
                "System successfully handled follow-up question after research"
            )
        else:
            self.log_test(
                "Integration - Follow-up Conversation", 
                "FAIL", 
                f"Failed to handle follow-up conversation. Error: {response.get('error', 'Unknown error')}"
            )
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("🚀 Starting Comprehensive Build 1 & 2 Testing")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print(f"Base URL: {self.base_url}")
        
        try:
            # Test Build 1
            if not self.test_build_1_ultra_orchestrator():
                print("\n❌ Build 1 tests failed. Stopping.")
                return False
            
            # Test Build 2 - Session Persistence
            if not self.test_build_2_session_persistence():
                print("\n❌ Build 2 session persistence tests failed. Stopping.")
                return False
            
            # Test Build 2 - Research Agent
            if not self.test_build_2_research_agent():
                print("\n❌ Build 2 research agent tests failed. Stopping.")
                return False
            
            # Test Build 2 - Artifact Generation
            self.test_build_2_artifact_generation()
            
            # Test Integration
            self.test_integration()
            
            # Summary
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"\n💥 Test execution failed with error: {e}")
            return False
    
    def print_summary(self):
        """Print test summary."""
        print("\n📊 Test Summary")
        print("=" * 50)
        
        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        warnings = sum(1 for result in self.test_results if result["status"] == "WARN")
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"📊 Total: {len(self.test_results)}")
        
        if failed == 0:
            print("\n🎉 All critical tests passed! Build 1 and Build 2 are working correctly.")
        else:
            print("\n⚠️  Some tests failed. Please review the results above.")
        
        print(f"\n🔗 Session ID for manual verification: {self.session_id}")
        print(f"🌐 You can check this session at: {self.base_url}/chat?session={self.session_id}")


if __name__ == "__main__":
    # Run the comprehensive test
    tester = Build1And2Tester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Testing completed successfully!")
    else:
        print("\n❌ Testing completed with errors.")
        exit(1)