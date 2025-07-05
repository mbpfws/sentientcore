#!/usr/bin/env python3
"""
Build 2 Complete System Test

This script tests the complete Build 2 implementation including:
- Session persistence across conversations
- Research delegation and artifact generation
- Enhanced state management
- API endpoints for session management
- Research artifact download functionality
"""

import asyncio
import aiohttp
import json
import time
import os
from typing import Dict, Any, List

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"test_session_{int(time.time())}"

class Build2SystemTester:
    """Comprehensive tester for Build 2 functionality"""
    
    def __init__(self):
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    async def test_session_persistence(self) -> bool:
        """Test session persistence functionality"""
        print("\n🔄 Testing Session Persistence...")
        
        try:
            # Test 1: Send first message with session ID
            first_message = "Hello, I'm testing Build 2 session persistence. Please remember this conversation."
            
            async with self.session.post(
                f"{BASE_URL}/chat/message/json",
                json={
                    "message": first_message,
                    "session_id": TEST_SESSION_ID
                }
            ) as response:
                if response.status != 200:
                    self.log_test("Session Creation", False, f"HTTP {response.status}")
                    return False
                
                data = await response.json()
                if not data.get("success"):
                    self.log_test("Session Creation", False, "API returned success=false")
                    return False
                
                response_session_id = data.get("data", {}).get("session_id")
                if response_session_id != TEST_SESSION_ID:
                    self.log_test("Session Creation", False, f"Session ID mismatch: {response_session_id}")
                    return False
                
                self.log_test("Session Creation", True, f"Session {TEST_SESSION_ID} created")
            
            # Test 2: Send follow-up message with same session ID
            await asyncio.sleep(2)  # Brief pause
            
            second_message = "Do you remember what I just told you about testing Build 2?"
            
            async with self.session.post(
                f"{BASE_URL}/chat/message/json",
                json={
                    "message": second_message,
                    "session_id": TEST_SESSION_ID
                }
            ) as response:
                if response.status != 200:
                    self.log_test("Session Continuity", False, f"HTTP {response.status}")
                    return False
                
                data = await response.json()
                response_content = data.get("data", {}).get("content", "")
                
                # Check if response indicates memory of previous conversation
                memory_indicators = ["remember", "testing", "Build 2", "persistence", "conversation"]
                has_memory = any(indicator.lower() in response_content.lower() for indicator in memory_indicators)
                
                self.log_test("Session Continuity", has_memory, f"Response: {response_content[:100]}...")
            
            # Test 3: Retrieve conversation history
            async with self.session.get(
                f"{BASE_URL}/chat/history",
                params={"session_id": TEST_SESSION_ID}
            ) as response:
                if response.status != 200:
                    self.log_test("History Retrieval", False, f"HTTP {response.status}")
                    return False
                
                data = await response.json()
                messages = data.get("data", {}).get("messages", [])
                
                # Should have at least 4 messages (2 user + 2 assistant)
                has_history = len(messages) >= 4
                self.log_test("History Retrieval", has_history, f"Found {len(messages)} messages")
            
            return True
            
        except Exception as e:
            self.log_test("Session Persistence", False, f"Exception: {str(e)}")
            return False
    
    async def test_research_delegation(self) -> bool:
        """Test research delegation and artifact generation"""
        print("\n🔬 Testing Research Delegation...")
        
        try:
            # Test research request
            research_query = "Please conduct a Deep Research on the latest developments in AI agent frameworks and multi-agent systems in 2024"
            
            async with self.session.post(
                f"{BASE_URL}/chat/message/json",
                json={
                    "message": research_query,
                    "research_mode": "deep",
                    "session_id": TEST_SESSION_ID
                }
            ) as response:
                if response.status != 200:
                    self.log_test("Research Request", False, f"HTTP {response.status}")
                    return False
                
                data = await response.json()
                response_content = data.get("data", {}).get("content", "")
                
                # Check if response indicates research was conducted
                research_indicators = ["research", "findings", "analysis", "sources", "comprehensive"]
                is_research_response = any(indicator.lower() in response_content.lower() for indicator in research_indicators)
                
                self.log_test("Research Request", is_research_response, f"Response length: {len(response_content)}")
            
            # Wait for potential artifact generation
            await asyncio.sleep(5)
            
            # Test artifact listing
            async with self.session.get(f"{BASE_URL}/chat/research/artifacts") as response:
                if response.status != 200:
                    self.log_test("Artifact Listing", False, f"HTTP {response.status}")
                    return False
                
                data = await response.json()
                artifacts = data.get("data", {}).get("artifacts", [])
                
                has_artifacts = len(artifacts) > 0
                self.log_test("Artifact Generation", has_artifacts, f"Found {len(artifacts)} artifacts")
                
                # Test artifact download if available
                if artifacts:
                    first_artifact = artifacts[0]
                    filename = first_artifact.get("filename")
                    
                    async with self.session.get(f"{BASE_URL}/chat/download/research/{filename}") as download_response:
                        download_success = download_response.status == 200
                        self.log_test("Artifact Download", download_success, f"Downloaded {filename}")
            
            return True
            
        except Exception as e:
            self.log_test("Research Delegation", False, f"Exception: {str(e)}")
            return False
    
    async def test_session_management_endpoints(self) -> bool:
        """Test session management API endpoints"""
        print("\n📊 Testing Session Management Endpoints...")
        
        try:
            # Test session listing
            async with self.session.get(f"{BASE_URL}/chat/sessions") as response:
                if response.status != 200:
                    self.log_test("Session Listing", False, f"HTTP {response.status}")
                    return False
                
                data = await response.json()
                sessions = data.get("data", {}).get("sessions", [])
                
                # Should find our test session
                test_session_found = any(s.get("session_id") == TEST_SESSION_ID for s in sessions)
                self.log_test("Session Listing", test_session_found, f"Found {len(sessions)} sessions")
            
            # Test session statistics
            async with self.session.get(f"{BASE_URL}/chat/sessions/{TEST_SESSION_ID}/stats") as response:
                if response.status != 200:
                    self.log_test("Session Stats", False, f"HTTP {response.status}")
                    return False
                
                data = await response.json()
                stats = data.get("data", {})
                
                has_stats = bool(stats and "message_count" in stats)
                self.log_test("Session Stats", has_stats, f"Stats: {stats}")
            
            return True
            
        except Exception as e:
            self.log_test("Session Management", False, f"Exception: {str(e)}")
            return False
    
    async def test_enhanced_state_management(self) -> bool:
        """Test enhanced state management features"""
        print("\n🔧 Testing Enhanced State Management...")
        
        try:
            # Test complex conversation with multiple turns
            messages = [
                "Let's test the enhanced state management. Can you help me plan a simple web application?",
                "I want to build a todo list app with React and FastAPI. What would be the architecture?",
                "How would you handle user authentication in this setup?"
            ]
            
            for i, message in enumerate(messages):
                async with self.session.post(
                    f"{BASE_URL}/chat/message/json",
                    json={
                        "message": message,
                        "session_id": TEST_SESSION_ID
                    }
                ) as response:
                    if response.status != 200:
                        self.log_test(f"State Management Turn {i+1}", False, f"HTTP {response.status}")
                        return False
                    
                    data = await response.json()
                    response_content = data.get("data", {}).get("content", "")
                    
                    # Check for contextual awareness
                    has_context = len(response_content) > 50  # Basic check for substantial response
                    self.log_test(f"State Management Turn {i+1}", has_context, f"Response length: {len(response_content)}")
                
                await asyncio.sleep(1)  # Brief pause between messages
            
            # Final history check
            async with self.session.get(
                f"{BASE_URL}/chat/history",
                params={"session_id": TEST_SESSION_ID}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    total_messages = len(data.get("data", {}).get("messages", []))
                    
                    # Should have accumulated many messages by now
                    has_accumulated_state = total_messages >= 10
                    self.log_test("State Accumulation", has_accumulated_state, f"Total messages: {total_messages}")
            
            return True
            
        except Exception as e:
            self.log_test("Enhanced State Management", False, f"Exception: {str(e)}")
            return False
    
    async def cleanup_test_session(self):
        """Clean up test session"""
        print("\n🧹 Cleaning up test session...")
        
        try:
            async with self.session.delete(f"{BASE_URL}/chat/sessions/{TEST_SESSION_ID}") as response:
                if response.status == 200:
                    self.log_test("Session Cleanup", True, f"Deleted session {TEST_SESSION_ID}")
                else:
                    self.log_test("Session Cleanup", False, f"HTTP {response.status}")
        except Exception as e:
            self.log_test("Session Cleanup", False, f"Exception: {str(e)}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("🎯 BUILD 2 COMPLETE SYSTEM TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")
        
        print("\n" + "="*60)
        
        # Save results to file
        results_file = "build2_test_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": (passed_tests/total_tests)*100
                },
                "test_results": self.test_results,
                "test_session_id": TEST_SESSION_ID,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")

async def main():
    """Main test execution"""
    print("🚀 Starting Build 2 Complete System Test")
    print(f"🔗 Testing against: {BASE_URL}")
    print(f"🆔 Test Session ID: {TEST_SESSION_ID}")
    
    async with Build2SystemTester() as tester:
        # Run all test suites
        await tester.test_session_persistence()
        await tester.test_research_delegation()
        await tester.test_session_management_endpoints()
        await tester.test_enhanced_state_management()
        
        # Cleanup
        await tester.cleanup_test_session()
        
        # Print summary
        tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())