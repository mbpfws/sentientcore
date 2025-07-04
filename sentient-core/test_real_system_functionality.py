#!/usr/bin/env python3
"""
Real System Functionality Test
Tests actual agent interactions, conversations, research capabilities, and workflow execution
"""

import asyncio
import requests
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class RealSystemTester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.test_results = []
        
    def log_result(self, test_name: str, status: str, details: str = "", data: Any = None):
        """Log test results with timestamp"""
        result = {
            "test_name": test_name,
            "status": status,
            "details": details,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "ℹ️"
        print(f"{status_symbol} {test_name}: {status}")
        if details:
            print(f"   {details}")
        if data and isinstance(data, dict) and len(str(data)) < 200:
            print(f"   Data: {data}")
    
    def test_conversation_with_orchestrator(self):
        """Test real conversation flow with orchestrator"""
        print("\n🗣️  Testing Real Conversation Flow...")
        
        conversation_tests = [
            {
                "name": "Project Planning Request",
                "message": "I want to build a Vietnamese language learning app with AI tutoring features. Can you help me plan this project?",
                "expected_elements": ["plan", "task", "agent"]
            },
            {
                "name": "Technical Research Request", 
                "message": "Research the best AI models for Vietnamese language processing and speech recognition",
                "expected_elements": ["research", "model", "vietnamese"]
            },
            {
                "name": "Architecture Question",
                "message": "What would be the best architecture for a real-time language learning app with voice interaction?",
                "expected_elements": ["architecture", "real-time", "voice"]
            }
        ]
        
        for test in conversation_tests:
            try:
                chat_data = {
                    "message": test["message"],
                    "workflow_mode": "intelligent"
                }
                
                response = requests.post(
                    f"{self.backend_url}/api/chat/message",
                    json=chat_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = str(result).lower()
                    
                    # Check if response contains expected elements
                    found_elements = [elem for elem in test["expected_elements"] if elem in response_text]
                    
                    if len(found_elements) >= 2:  # At least 2 expected elements found
                        self.log_result(
                            f"Conversation: {test['name']}",
                            "PASS",
                            f"Found {len(found_elements)}/{len(test['expected_elements'])} expected elements",
                            {"response_length": len(str(result)), "found_elements": found_elements}
                        )
                    else:
                        self.log_result(
                            f"Conversation: {test['name']}",
                            "PARTIAL",
                            f"Only found {len(found_elements)}/{len(test['expected_elements'])} expected elements",
                            {"response": str(result)[:200] + "..."}
                        )
                else:
                    self.log_result(
                        f"Conversation: {test['name']}",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:100]}"
                    )
                    
            except Exception as e:
                self.log_result(
                    f"Conversation: {test['name']}",
                    "FAIL",
                    f"Exception: {str(e)}"
                )
    
    def test_agent_endpoints(self):
        """Test agent-specific endpoints"""
        print("\n🤖 Testing Agent Endpoints...")
        
        agent_tests = [
            {
                "endpoint": "/api/agents",
                "method": "GET",
                "name": "List Available Agents"
            },
            {
                "endpoint": "/api/agents/research",
                "method": "POST",
                "name": "Research Agent",
                "data": {
                    "query": "Best practices for Vietnamese NLP models",
                    "research_type": "technical"
                }
            },
            {
                "endpoint": "/api/agents/architect",
                "method": "POST", 
                "name": "Architect Agent",
                "data": {
                    "project_description": "Language learning mobile app",
                    "requirements": ["real-time voice", "AI tutoring", "progress tracking"]
                }
            }
        ]
        
        for test in agent_tests:
            try:
                if test["method"] == "GET":
                    response = requests.get(f"{self.backend_url}{test['endpoint']}", timeout=15)
                else:
                    response = requests.post(
                        f"{self.backend_url}{test['endpoint']}",
                        json=test.get("data", {}),
                        timeout=30
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    self.log_result(
                        f"Agent: {test['name']}",
                        "PASS",
                        f"Response received: {len(str(result))} chars",
                        {"status_code": response.status_code}
                    )
                elif response.status_code == 404:
                    self.log_result(
                        f"Agent: {test['name']}",
                        "NOT_IMPLEMENTED",
                        "Endpoint not yet implemented"
                    )
                else:
                    self.log_result(
                        f"Agent: {test['name']}",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:100]}"
                    )
                    
            except Exception as e:
                self.log_result(
                    f"Agent: {test['name']}",
                    "FAIL",
                    f"Exception: {str(e)}"
                )
    
    def test_workflow_endpoints(self):
        """Test workflow execution endpoints"""
        print("\n🔄 Testing Workflow Endpoints...")
        
        workflow_tests = [
            {
                "endpoint": "/api/workflows",
                "method": "GET",
                "name": "List Available Workflows"
            },
            {
                "endpoint": "/api/workflows/research",
                "method": "POST",
                "name": "Research Workflow",
                "data": {
                    "topic": "Vietnamese speech recognition APIs",
                    "depth": "comprehensive"
                }
            },
            {
                "endpoint": "/api/workflows/design",
                "method": "POST",
                "name": "Design Workflow",
                "data": {
                    "project_type": "mobile_app",
                    "features": ["voice_interaction", "ai_tutoring", "progress_tracking"]
                }
            }
        ]
        
        for test in workflow_tests:
            try:
                if test["method"] == "GET":
                    response = requests.get(f"{self.backend_url}{test['endpoint']}", timeout=15)
                else:
                    response = requests.post(
                        f"{self.backend_url}{test['endpoint']}",
                        json=test.get("data", {}),
                        timeout=45  # Workflows might take longer
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    self.log_result(
                        f"Workflow: {test['name']}",
                        "PASS",
                        f"Workflow executed: {len(str(result))} chars response",
                        {"status_code": response.status_code}
                    )
                elif response.status_code == 404:
                    self.log_result(
                        f"Workflow: {test['name']}",
                        "NOT_IMPLEMENTED",
                        "Workflow endpoint not yet implemented"
                    )
                else:
                    self.log_result(
                        f"Workflow: {test['name']}",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:100]}"
                    )
                    
            except Exception as e:
                self.log_result(
                    f"Workflow: {test['name']}",
                    "FAIL",
                    f"Exception: {str(e)}"
                )
    
    def test_direct_agent_imports(self):
        """Test direct agent imports and functionality"""
        print("\n🔧 Testing Direct Agent Imports...")
        
        try:
            # Test ultra orchestrator import
            from core.agents.ultra_orchestrator import UltraOrchestrator
            self.log_result(
                "Direct Import: UltraOrchestrator",
                "PASS",
                "Successfully imported UltraOrchestrator class"
            )
            
            # Test research agent import
            from core.agents.research_agent import ResearchAgent
            self.log_result(
                "Direct Import: ResearchAgent",
                "PASS",
                "Successfully imported ResearchAgent class"
            )
            
            # Test workflow graph import
            from graphs.research_graph import research_app
            self.log_result(
                "Direct Import: ResearchWorkflow",
                "PASS",
                "Successfully imported research workflow graph"
            )
            
        except ImportError as e:
            self.log_result(
                "Direct Import: Agents",
                "FAIL",
                f"Import error: {str(e)}"
            )
        except Exception as e:
            self.log_result(
                "Direct Import: Agents",
                "FAIL",
                f"Unexpected error: {str(e)}"
            )
    
    def test_system_integration(self):
        """Test end-to-end system integration"""
        print("\n🔗 Testing System Integration...")
        
        try:
            # Test a complete workflow: user request -> orchestrator -> agent -> response
            integration_request = {
                "message": "I need help designing a Vietnamese language learning app. Please research the best approaches and create an initial architecture plan.",
                "workflow_mode": "comprehensive"
            }
            
            response = requests.post(
                f"{self.backend_url}/api/chat/message",
                json=integration_request,
                timeout=60  # Allow more time for complex processing
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = str(result).lower()
                
                # Check for integration indicators
                integration_indicators = [
                    "research", "architecture", "plan", "vietnamese", 
                    "language", "app", "design", "approach"
                ]
                
                found_indicators = [ind for ind in integration_indicators if ind in response_text]
                
                if len(found_indicators) >= 4:
                    self.log_result(
                        "System Integration: Full Workflow",
                        "PASS",
                        f"Complex request processed successfully. Found {len(found_indicators)} relevant indicators.",
                        {"response_length": len(str(result)), "indicators_found": len(found_indicators)}
                    )
                else:
                    self.log_result(
                        "System Integration: Full Workflow",
                        "PARTIAL",
                        f"Request processed but limited integration. Found {len(found_indicators)} indicators.",
                        {"response": str(result)[:300] + "..."}
                    )
            else:
                self.log_result(
                    "System Integration: Full Workflow",
                    "FAIL",
                    f"HTTP {response.status_code}: {response.text[:100]}"
                )
                
        except Exception as e:
            self.log_result(
                "System Integration: Full Workflow",
                "FAIL",
                f"Exception: {str(e)}"
            )
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("🎯 REAL SYSTEM FUNCTIONALITY TEST REPORT")
        print("="*80)
        
        # Count results by status
        status_counts = {}
        for result in self.test_results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n📊 Test Summary:")
        for status, count in status_counts.items():
            emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️" if status == "PARTIAL" else "ℹ️"
            print(f"   {emoji} {status}: {count} tests")
        
        print(f"\n📋 Total Tests Run: {len(self.test_results)}")
        
        # Show key findings
        print("\n🔍 Key Findings:")
        
        # Working features
        working_features = [r for r in self.test_results if r["status"] == "PASS"]
        if working_features:
            print("\n✅ Working Features:")
            for feature in working_features[:5]:  # Show top 5
                print(f"   • {feature['test_name']}")
        
        # Issues found
        issues = [r for r in self.test_results if r["status"] == "FAIL"]
        if issues:
            print("\n❌ Issues Found:")
            for issue in issues[:5]:  # Show top 5
                print(f"   • {issue['test_name']}: {issue['details'][:60]}...")
        
        # Partially working
        partial = [r for r in self.test_results if r["status"] == "PARTIAL"]
        if partial:
            print("\n⚠️ Partially Working:")
            for p in partial:
                print(f"   • {p['test_name']}: {p['details'][:60]}...")
        
        return {
            "total_tests": len(self.test_results),
            "status_counts": status_counts,
            "detailed_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_all_tests(self):
        """Run comprehensive system functionality tests"""
        print("🚀 Starting Real System Functionality Tests...")
        print("="*80)
        
        # Test direct imports first
        self.test_direct_agent_imports()
        
        # Test conversation capabilities
        self.test_conversation_with_orchestrator()
        
        # Test agent endpoints
        self.test_agent_endpoints()
        
        # Test workflow endpoints
        self.test_workflow_endpoints()
        
        # Test system integration
        self.test_system_integration()
        
        # Generate and return summary
        return self.generate_summary_report()

if __name__ == "__main__":
    tester = RealSystemTester()
    summary = tester.run_all_tests()
    
    # Save detailed results
    with open("real_system_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: real_system_test_results.json")
    print("\n🎉 Real system functionality testing completed!")