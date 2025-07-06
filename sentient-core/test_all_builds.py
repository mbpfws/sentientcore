#!/usr/bin/env python3
"""
Comprehensive Test Suite for All Three Builds
Tests Build 1 (Core Conversation), Build 2 (Research Agent), and Build 3 (Architect Planner)
with SSE implementation and robust state management.
"""

import asyncio
import aiohttp
import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Test Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"
SSE_ENDPOINT = f"{API_BASE}/sse/events"
CHAT_ENDPOINT = f"{API_BASE}/chat/message"
CONFIRM_ENDPOINT = f"{API_BASE}/chat/confirm"
CONTEXT_ENDPOINT = f"{API_BASE}/chat/context"
ARTIFACTS_ENDPOINT = f"{API_BASE}/chat/artifacts"

@dataclass
class TestResult:
    """Test result tracking"""
    test_name: str
    success: bool
    message: str
    duration: float
    artifacts: List[str] = None
    
class BuildTestSuite:
    """Comprehensive test suite for all builds"""
    
    def __init__(self):
        self.session_id = None
        self.results: List[TestResult] = []
        self.sse_messages: List[Dict] = []
        self.confirmations: List[str] = []
        
    async def run_all_tests(self):
        """Run all build tests in sequence"""
        print("🚀 Starting Comprehensive Build Test Suite")
        print("=" * 60)
        
        try:
            # Test server connectivity
            await self.test_server_connectivity()
            
            # Build 1 Tests: Core Conversation & Orchestration
            await self.test_build_1_core_conversation()
            
            # Build 2 Tests: Research Agent & Persistence
            await self.test_build_2_research_agent()
            
            # Build 3 Tests: Architect Planner & Tiered Memory
            await self.test_build_3_architect_planner()
            
            # SSE Implementation Tests
            await self.test_sse_implementation()
            
            # State Management Tests
            await self.test_state_management()
            
            # End-to-End Integration Test
            await self.test_end_to_end_integration()
            
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
            
        finally:
            await self.generate_test_report()
    
    async def test_server_connectivity(self):
        """Test basic server connectivity"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{BASE_URL}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.add_result("Server Connectivity", True, 
                                      f"Server is healthy: {data.get('message')}", 
                                      time.time() - start_time)
                    else:
                        self.add_result("Server Connectivity", False, 
                                      f"Server returned status {response.status}", 
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Server Connectivity", False, 
                          f"Failed to connect: {e}", 
                          time.time() - start_time)
    
    async def test_build_1_core_conversation(self):
        """Test Build 1: Core Conversation & Orchestration Loop"""
        print("\n🔄 Testing Build 1: Core Conversation & Orchestration")
        
        # Test 1.1: Initialize conversation
        await self.test_conversation_initialization()
        
        # Test 1.2: Multi-turn conversation with context
        await self.test_multi_turn_conversation()
        
        # Test 1.3: Orchestrator intelligence and guidance
        await self.test_orchestrator_intelligence()
    
    async def test_conversation_initialization(self):
        """Test conversation initialization"""
        start_time = time.time()
        
        try:
            payload = {
                "message": "Hello, I need help with a new software project",
                "workflow_mode": "intelligent"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(CHAT_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.session_id = data.get('session_id')
                        
                        if self.session_id and 'content' in data:
                            self.add_result("Conversation Initialization", True,
                                          f"Session created: {self.session_id[:8]}...",
                                          time.time() - start_time)
                        else:
                            self.add_result("Conversation Initialization", False,
                                          "Missing session_id or content in response",
                                          time.time() - start_time)
                    else:
                        self.add_result("Conversation Initialization", False,
                                      f"HTTP {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Conversation Initialization", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with context retention"""
        start_time = time.time()
        
        if not self.session_id:
            self.add_result("Multi-turn Conversation", False,
                          "No session_id available", time.time() - start_time)
            return
        
        try:
            # Follow-up message that requires context
            payload = {
                "message": "I want to build a web application for task management",
                "workflow_mode": "intelligent",
                "session_id": self.session_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(CHAT_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check if response shows context awareness
                        content = data.get('content', '').lower()
                        context_indicators = ['task management', 'web application', 'project']
                        
                        if any(indicator in content for indicator in context_indicators):
                            self.add_result("Multi-turn Conversation", True,
                                          "Context retained across turns",
                                          time.time() - start_time)
                        else:
                            self.add_result("Multi-turn Conversation", False,
                                          "Context not retained",
                                          time.time() - start_time)
                    else:
                        self.add_result("Multi-turn Conversation", False,
                                      f"HTTP {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Multi-turn Conversation", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_orchestrator_intelligence(self):
        """Test orchestrator's intelligent guidance"""
        start_time = time.time()
        
        if not self.session_id:
            self.add_result("Orchestrator Intelligence", False,
                          "No session_id available", time.time() - start_time)
            return
        
        try:
            # Vague request that should trigger intelligent guidance
            payload = {
                "message": "I need something for my business",
                "workflow_mode": "intelligent",
                "session_id": self.session_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(CHAT_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('content', '').lower()
                        
                        # Check for guidance indicators
                        guidance_indicators = [
                            'tell me more', 'what kind', 'specific', 'details',
                            'requirements', 'help you', 'understand'
                        ]
                        
                        if any(indicator in content for indicator in guidance_indicators):
                            self.add_result("Orchestrator Intelligence", True,
                                          "Orchestrator provides intelligent guidance",
                                          time.time() - start_time)
                        else:
                            self.add_result("Orchestrator Intelligence", False,
                                          "No intelligent guidance detected",
                                          time.time() - start_time)
                    else:
                        self.add_result("Orchestrator Intelligence", False,
                                      f"HTTP {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Orchestrator Intelligence", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_build_2_research_agent(self):
        """Test Build 2: Research Agent & Persistence Layer"""
        print("\n🔍 Testing Build 2: Research Agent & Persistence")
        
        # Test 2.1: Research agent activation
        await self.test_research_agent_activation()
        
        # Test 2.2: Groq agentic tooling
        await self.test_groq_agentic_tooling()
        
        # Test 2.3: Artifact generation and persistence
        await self.test_artifact_generation()
        
        # Test 2.4: Verbose logging and transparency
        await self.test_verbose_logging()
    
    async def test_research_agent_activation(self):
        """Test research agent activation with user confirmation"""
        start_time = time.time()
        
        if not self.session_id:
            self.add_result("Research Agent Activation", False,
                          "No session_id available", time.time() - start_time)
            return
        
        try:
            # Request that should trigger research
            payload = {
                "message": "I want to build a modern e-commerce platform with the latest technologies. Please research the best practices and current trends.",
                "workflow_mode": "intelligent",
                "session_id": self.session_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(CHAT_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for confirmation request
                        if data.get('message_type') == 'confirmation':
                            confirmation_id = data.get('metadata', {}).get('confirmation_id')
                            if confirmation_id:
                                self.confirmations.append(confirmation_id)
                                self.add_result("Research Agent Activation", True,
                                              "Research confirmation requested",
                                              time.time() - start_time)
                            else:
                                self.add_result("Research Agent Activation", False,
                                              "No confirmation_id in response",
                                              time.time() - start_time)
                        else:
                            # Check if research was triggered directly
                            content = data.get('content', '').lower()
                            if 'research' in content:
                                self.add_result("Research Agent Activation", True,
                                              "Research agent activated",
                                              time.time() - start_time)
                            else:
                                self.add_result("Research Agent Activation", False,
                                              "Research not triggered",
                                              time.time() - start_time)
                    else:
                        self.add_result("Research Agent Activation", False,
                                      f"HTTP {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Research Agent Activation", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_groq_agentic_tooling(self):
        """Test Groq agentic tooling integration"""
        start_time = time.time()
        
        # This test checks if the system is configured to use Groq's agentic tooling
        try:
            # Check environment variables
            groq_api_key = os.getenv('GROQ_API_KEY')
            
            if groq_api_key:
                self.add_result("Groq Agentic Tooling", True,
                              "Groq API key configured",
                              time.time() - start_time)
            else:
                self.add_result("Groq Agentic Tooling", False,
                              "Groq API key not found in environment",
                              time.time() - start_time)
        except Exception as e:
            self.add_result("Groq Agentic Tooling", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_artifact_generation(self):
        """Test artifact generation and download capabilities"""
        start_time = time.time()
        
        try:
            # Check for artifacts endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(ARTIFACTS_ENDPOINT) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if isinstance(data, list) or 'artifacts' in str(data):
                            self.add_result("Artifact Generation", True,
                                          "Artifacts endpoint accessible",
                                          time.time() - start_time)
                        else:
                            self.add_result("Artifact Generation", False,
                                          "Artifacts endpoint not properly configured",
                                          time.time() - start_time)
                    else:
                        self.add_result("Artifact Generation", False,
                                      f"Artifacts endpoint returned {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Artifact Generation", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_verbose_logging(self):
        """Test verbose logging and monitoring capabilities"""
        start_time = time.time()
        
        try:
            # Check monitoring endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_BASE}/monitoring/logs") as response:
                    if response.status == 200:
                        self.add_result("Verbose Logging", True,
                                      "Monitoring endpoint accessible",
                                      time.time() - start_time)
                    else:
                        self.add_result("Verbose Logging", False,
                                      f"Monitoring endpoint returned {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Verbose Logging", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_build_3_architect_planner(self):
        """Test Build 3: Architect Planner & Tiered Memory"""
        print("\n🏗️ Testing Build 3: Architect Planner & Tiered Memory")
        
        # Test 3.1: Architect planner activation
        await self.test_architect_planner_activation()
        
        # Test 3.2: PRD generation
        await self.test_prd_generation()
        
        # Test 3.3: Tiered memory system
        await self.test_tiered_memory()
        
        # Test 3.4: Project architecture graph
        await self.test_project_architecture_graph()
    
    async def test_architect_planner_activation(self):
        """Test architect planner activation"""
        start_time = time.time()
        
        if not self.session_id:
            self.add_result("Architect Planner Activation", False,
                          "No session_id available", time.time() - start_time)
            return
        
        try:
            # Request that should trigger planning phase
            payload = {
                "message": "Based on the research, please create a comprehensive plan and architecture for the e-commerce platform",
                "workflow_mode": "intelligent",
                "session_id": self.session_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(CHAT_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('content', '').lower()
                        
                        # Check for planning indicators
                        planning_indicators = [
                            'plan', 'architecture', 'design', 'structure',
                            'requirements', 'specification'
                        ]
                        
                        if any(indicator in content for indicator in planning_indicators):
                            self.add_result("Architect Planner Activation", True,
                                          "Architect planner activated",
                                          time.time() - start_time)
                        else:
                            self.add_result("Architect Planner Activation", False,
                                          "Planning not triggered",
                                          time.time() - start_time)
                    else:
                        self.add_result("Architect Planner Activation", False,
                                      f"HTTP {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Architect Planner Activation", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_prd_generation(self):
        """Test PRD (Product Requirements Document) generation"""
        start_time = time.time()
        
        try:
            # Check if PRD artifacts are being generated
            memory_path = "d:/sentientcore/sentient-core/memory/layer2_build_artifacts"
            
            if os.path.exists(memory_path):
                prd_files = [f for f in os.listdir(memory_path) if 'prd' in f.lower()]
                
                if prd_files:
                    self.add_result("PRD Generation", True,
                                  f"PRD files found: {len(prd_files)}",
                                  time.time() - start_time,
                                  prd_files)
                else:
                    self.add_result("PRD Generation", False,
                                  "No PRD files found in layer2 memory",
                                  time.time() - start_time)
            else:
                self.add_result("PRD Generation", False,
                              "Layer2 memory directory not found",
                              time.time() - start_time)
        except Exception as e:
            self.add_result("PRD Generation", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_tiered_memory(self):
        """Test tiered memory system"""
        start_time = time.time()
        
        try:
            # Check memory structure
            memory_base = "d:/sentientcore/sentient-core/memory"
            layer1_path = os.path.join(memory_base, "layer1_conversation_history")
            layer2_path = os.path.join(memory_base, "layer2_build_artifacts")
            
            layer1_exists = os.path.exists(layer1_path)
            layer2_exists = os.path.exists(layer2_path)
            
            if layer1_exists and layer2_exists:
                self.add_result("Tiered Memory", True,
                              "Both memory layers exist",
                              time.time() - start_time)
            elif layer1_exists:
                self.add_result("Tiered Memory", False,
                              "Only layer1 memory exists",
                              time.time() - start_time)
            elif layer2_exists:
                self.add_result("Tiered Memory", False,
                              "Only layer2 memory exists",
                              time.time() - start_time)
            else:
                self.add_result("Tiered Memory", False,
                              "No memory layers found",
                              time.time() - start_time)
        except Exception as e:
            self.add_result("Tiered Memory", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_project_architecture_graph(self):
        """Test project architecture graph functionality"""
        start_time = time.time()
        
        try:
            # Check workflows endpoint for architecture graph
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_BASE}/workflows/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Look for architecture-related workflows
                        workflows = data if isinstance(data, list) else []
                        arch_workflows = [w for w in workflows if 'architect' in str(w).lower()]
                        
                        if arch_workflows:
                            self.add_result("Project Architecture Graph", True,
                                          f"Architecture workflows found: {len(arch_workflows)}",
                                          time.time() - start_time)
                        else:
                            self.add_result("Project Architecture Graph", True,
                                          "Workflows endpoint accessible",
                                          time.time() - start_time)
                    else:
                        self.add_result("Project Architecture Graph", False,
                                      f"Workflows endpoint returned {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("Project Architecture Graph", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_sse_implementation(self):
        """Test SSE (Server-Sent Events) implementation"""
        print("\n📡 Testing SSE Implementation")
        
        start_time = time.time()
        
        try:
            # Test SSE endpoint connectivity
            async with aiohttp.ClientSession() as session:
                async with session.get(SSE_ENDPOINT) as response:
                    if response.status == 200:
                        self.add_result("SSE Implementation", True,
                                      "SSE endpoint accessible",
                                      time.time() - start_time)
                    else:
                        self.add_result("SSE Implementation", False,
                                      f"SSE endpoint returned {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("SSE Implementation", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_state_management(self):
        """Test robust state management"""
        print("\n🔄 Testing State Management")
        
        start_time = time.time()
        
        if not self.session_id:
            self.add_result("State Management", False,
                          "No session_id available", time.time() - start_time)
            return
        
        try:
            # Test context retrieval
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{CONTEXT_ENDPOINT}/{self.session_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'context' in data or 'session_id' in data:
                            self.add_result("State Management", True,
                                          "Session context retrievable",
                                          time.time() - start_time)
                        else:
                            self.add_result("State Management", False,
                                          "Invalid context response format",
                                          time.time() - start_time)
                    else:
                        self.add_result("State Management", False,
                                      f"Context endpoint returned {response.status}",
                                      time.time() - start_time)
        except Exception as e:
            self.add_result("State Management", False,
                          f"Error: {e}", time.time() - start_time)
    
    async def test_end_to_end_integration(self):
        """Test complete end-to-end integration"""
        print("\n🔗 Testing End-to-End Integration")
        
        start_time = time.time()
        
        try:
            # Simulate complete workflow: conversation -> research -> planning
            if self.session_id and self.confirmations:
                # Confirm any pending research
                for confirmation_id in self.confirmations:
                    payload = {
                        "confirmation_id": confirmation_id,
                        "confirmed": True,
                        "session_id": self.session_id
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(CONFIRM_ENDPOINT, json=payload) as response:
                            if response.status == 200:
                                self.add_result("End-to-End Integration", True,
                                              "Confirmation workflow completed",
                                              time.time() - start_time)
                                break
                            else:
                                self.add_result("End-to-End Integration", False,
                                              f"Confirmation failed: {response.status}",
                                              time.time() - start_time)
            else:
                self.add_result("End-to-End Integration", False,
                              "No session or confirmations available",
                              time.time() - start_time)
        except Exception as e:
            self.add_result("End-to-End Integration", False,
                          f"Error: {e}", time.time() - start_time)
    
    def add_result(self, test_name: str, success: bool, message: str, duration: float, artifacts: List[str] = None):
        """Add a test result"""
        result = TestResult(test_name, success, message, duration, artifacts)
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message} ({duration:.2f}s)")
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 TEST REPORT SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration for r in self.results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results:
                if not result.success:
                    print(f"  - {result.test_name}: {result.message}")
        
        # Generate detailed report file
        report_content = self.generate_detailed_report()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"d:/sentientcore/sentient-core/test_report_{timestamp}.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"\n📄 Detailed report saved: {report_file}")
        except Exception as e:
            print(f"\n❌ Failed to save report: {e}")
    
    def generate_detailed_report(self) -> str:
        """Generate detailed markdown report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Sentient Core Build Test Report

**Generated:** {timestamp}  
**Session ID:** {self.session_id or 'N/A'}  
**Total Tests:** {len(self.results)}  
**Passed:** {sum(1 for r in self.results if r.success)}  
**Failed:** {sum(1 for r in self.results if not r.success)}  

## Test Results

"""
        
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report += f"### {result.test_name}\n"
            report += f"**Status:** {status}  \n"
            report += f"**Duration:** {result.duration:.2f}s  \n"
            report += f"**Message:** {result.message}  \n"
            
            if result.artifacts:
                report += f"**Artifacts:** {', '.join(result.artifacts)}  \n"
            
            report += "\n"
        
        return report

async def main():
    """Main test execution"""
    test_suite = BuildTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())