#!/usr/bin/env python3
"""
Comprehensive System Test for SentientCore
Tests memory functionality, orchestrator conversation flow, and development stage assessment
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

class SystemTester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3001"
        self.test_results = {
            "backend_tests": {},
            "frontend_tests": {},
            "memory_tests": {},
            "orchestrator_tests": {},
            "development_stage": {}
        }
    
    def log_test(self, category: str, test_name: str, status: str, details: str = ""):
        """Log test results"""
        timestamp = datetime.now().isoformat()
        self.test_results[category][test_name] = {
            "status": status,
            "details": details,
            "timestamp": timestamp
        }
        print(f"[{timestamp}] {category}.{test_name}: {status}")
        if details:
            print(f"  Details: {details}")
    
    def test_backend_health(self):
        """Test backend server health"""
        try:
            response = requests.get(f"{self.backend_url}/", timeout=5)
            if response.status_code == 200:
                self.log_test("backend_tests", "health_check", "PASS", f"Status: {response.status_code}")
                return True
            else:
                self.log_test("backend_tests", "health_check", "FAIL", f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("backend_tests", "health_check", "FAIL", str(e))
            return False
    
    def test_frontend_health(self):
        """Test frontend server health"""
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                self.log_test("frontend_tests", "health_check", "PASS", f"Status: {response.status_code}")
                return True
            else:
                self.log_test("frontend_tests", "health_check", "FAIL", f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("frontend_tests", "health_check", "FAIL", str(e))
            return False
    
    def test_memory_storage(self):
        """Test memory storage functionality"""
        try:
            # Test memory storage endpoint with correct API format
            memory_data = {
                "layer": "conversation_history",
                "memory_type": "conversation",
                "content": "Test memory for system validation",
                "tags": ["test", "system_validation"],
                "metadata": {
                    "source": "system_test",
                    "importance": "high"
                }
            }
            
            response = requests.post(
                f"{self.backend_url}/api/core-services/memory/store",
                json=memory_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                memory_id = result.get("memory_id")
                self.log_test("memory_tests", "store_memory", "PASS", f"Memory ID: {memory_id}")
                return memory_id
            else:
                self.log_test("memory_tests", "store_memory", "FAIL", f"Status: {response.status_code}, Response: {response.text}")
                return None
        except Exception as e:
            self.log_test("memory_tests", "store_memory", "FAIL", str(e))
            return None
    
    def test_memory_retrieval(self):
        """Test memory retrieval functionality"""
        try:
            # Test memory retrieval endpoint
            query_data = {
                "query": "test memory",
                "limit": 5
            }
            
            response = requests.post(
                f"{self.backend_url}/api/core-services/memory/retrieve",
                json=query_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                memories = result.get("memories", [])
                self.log_test("memory_tests", "retrieve_memory", "PASS", f"Retrieved {len(memories)} memories")
                return memories
            else:
                self.log_test("memory_tests", "retrieve_memory", "FAIL", f"Status: {response.status_code}, Response: {response.text}")
                return []
        except Exception as e:
            self.log_test("memory_tests", "retrieve_memory", "FAIL", str(e))
            return []
    
    def test_memory_stats(self):
        """Test memory statistics endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/api/core-services/memory/stats", timeout=5)
            
            if response.status_code == 200:
                stats = response.json()
                self.log_test("memory_tests", "memory_stats", "PASS", f"Stats: {stats}")
                return stats
            else:
                self.log_test("memory_tests", "memory_stats", "FAIL", f"Status: {response.status_code}")
                return None
        except Exception as e:
            self.log_test("memory_tests", "memory_stats", "FAIL", str(e))
            return None
    
    def test_orchestrator_conversation(self):
        """Test conversation flow with orchestrator"""
        try:
            # Test chat endpoint with orchestrator
            chat_data = {
                "message": "Hello, can you help me understand the current system capabilities?",
                "workflow_mode": "intelligent"
            }
            
            response = requests.post(
                f"{self.backend_url}/api/chat/message",
                json=chat_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.log_test("orchestrator_tests", "conversation_flow", "PASS", f"Response received: {len(str(result))} chars")
                return result
            else:
                self.log_test("orchestrator_tests", "conversation_flow", "FAIL", f"Status: {response.status_code}, Response: {response.text}")
                return None
        except Exception as e:
            self.log_test("orchestrator_tests", "conversation_flow", "FAIL", str(e))
            return None
    
    def assess_development_stage(self):
        """Assess current development stage based on available endpoints and functionality"""
        
        # Check available endpoints
        endpoints_to_check = [
            "/",  # Health check
            "/api/core-services/memory/store",  # Memory storage
            "/api/core-services/memory/retrieve",  # Memory retrieval
            "/api/core-services/memory/stats",  # Memory stats
            "/api/chat/message",  # Chat/orchestrator
            "/api/agents",  # Agent endpoints
            "/api/workflows",  # Workflow endpoints
        ]
        
        available_endpoints = []
        for endpoint in endpoints_to_check:
            try:
                response = requests.get(f"{self.backend_url}{endpoint}", timeout=5)
                if response.status_code in [200, 405]:  # 405 means method not allowed but endpoint exists
                    available_endpoints.append(endpoint)
            except:
                pass
        
        # Map to development phases (based on the 25 planned files)
        phase_mapping = {
            "Phase 1 - Core Setup": ["/"],
            "Phase 2 - Core Services": ["/api/core-services/memory/store", "/api/core-services/memory/retrieve", "/api/core-services/memory/stats"],
            "Phase 3 - Agent Framework": ["/api/agents"],
            "Phase 4 - Orchestrator": ["/api/chat/message"],
            "Phase 5 - Workflows": ["/api/workflows"],
        }
        
        completed_phases = []
        for phase, required_endpoints in phase_mapping.items():
            if all(endpoint in available_endpoints for endpoint in required_endpoints):
                completed_phases.append(phase)
        
        self.log_test("development_stage", "endpoint_analysis", "INFO", 
                     f"Available endpoints: {available_endpoints}")
        self.log_test("development_stage", "phase_completion", "INFO", 
                     f"Completed phases: {completed_phases}")
        
        return {
            "available_endpoints": available_endpoints,
            "completed_phases": completed_phases,
            "total_phases": len(phase_mapping)
        }
    
    def run_all_tests(self):
        """Run all system tests"""
        print("=" * 60)
        print("SentientCore System Test Suite")
        print("=" * 60)
        
        # Test backend health
        print("\n1. Testing Backend Health...")
        backend_healthy = self.test_backend_health()
        
        # Test frontend health
        print("\n2. Testing Frontend Health...")
        frontend_healthy = self.test_frontend_health()
        
        if backend_healthy:
            # Test memory functionality
            print("\n3. Testing Memory Functionality...")
            memory_id = self.test_memory_storage()
            memories = self.test_memory_retrieval()
            stats = self.test_memory_stats()
            
            # Test orchestrator conversation
            print("\n4. Testing Orchestrator Conversation...")
            conversation_result = self.test_orchestrator_conversation()
            
            # Assess development stage
            print("\n5. Assessing Development Stage...")
            stage_assessment = self.assess_development_stage()
        
        # Generate summary report
        print("\n" + "=" * 60)
        print("TEST SUMMARY REPORT")
        print("=" * 60)
        
        for category, tests in self.test_results.items():
            print(f"\n{category.upper()}:")
            for test_name, result in tests.items():
                status_symbol = "✓" if result["status"] == "PASS" else "✗" if result["status"] == "FAIL" else "ℹ"
                print(f"  {status_symbol} {test_name}: {result['status']}")
                if result["details"]:
                    print(f"    {result['details']}")
        
        return self.test_results

if __name__ == "__main__":
    tester = SystemTester()
    results = tester.run_all_tests()
    
    # Save results to file
    with open("system_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: system_test_results.json")