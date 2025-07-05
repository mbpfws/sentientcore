#!/usr/bin/env python3
"""
Sentient Core - Frontend & Backend Testing Guide

This script provides comprehensive testing examples for both the frontend and backend services.

Running Services:
- Backend API: http://localhost:8000 (FastAPI with Uvicorn)
- Frontend: http://localhost:3000 (Next.js)

Available Endpoints and Testing Scenarios
"""

import requests
import json
from typing import Dict, Any
import time

# Service URLs
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

class SentientCoreTestSuite:
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.frontend_url = FRONTEND_URL
        self.session = requests.Session()
        
    def test_backend_health(self):
        """Test backend health endpoint"""
        try:
            response = self.session.get(f"{self.backend_url}/health")
            print(f"✓ Backend Health: {response.status_code} - {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"✗ Backend Health Failed: {e}")
            return False
    
    def test_frontend_health(self):
        """Test frontend accessibility"""
        try:
            response = self.session.get(self.frontend_url)
            print(f"✓ Frontend Accessible: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"✗ Frontend Access Failed: {e}")
            return False
    
    def test_chat_endpoint(self, message: str = "Hello, can you help me with a simple task?"):
        """Test the main chat endpoint"""
        try:
            payload = {
                "message": message,
                "conversation_id": "test-conversation-001",
                "user_id": "test-user"
            }
            
            response = self.session.post(
                f"{self.backend_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"\n=== Chat Test ===")
            print(f"Input: {message}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result.get('response', 'No response field')}")
                print(f"Conversation ID: {result.get('conversation_id', 'N/A')}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Chat Test Failed: {e}")
            return False
    
    def test_research_endpoint(self, query: str = "What are the latest trends in AI development?"):
        """Test the research agent endpoint"""
        try:
            payload = {
                "query": query,
                "research_type": "general",
                "max_results": 5
            }
            
            response = self.session.post(
                f"{self.backend_url}/api/research",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"\n=== Research Test ===")
            print(f"Query: {query}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Results found: {len(result.get('results', []))}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Research Test Failed: {e}")
            return False
    
    def test_memory_endpoint(self):
        """Test memory management endpoints"""
        try:
            # Test storing memory
            store_payload = {
                "conversation_id": "test-conversation-001",
                "content": "User prefers Python for backend development",
                "memory_type": "preference"
            }
            
            store_response = self.session.post(
                f"{self.backend_url}/api/memory/store",
                json=store_payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"\n=== Memory Store Test ===")
            print(f"Status: {store_response.status_code}")
            
            if store_response.status_code == 200:
                # Test retrieving memory
                retrieve_response = self.session.get(
                    f"{self.backend_url}/api/memory/retrieve/test-conversation-001"
                )
                
                print(f"Retrieve Status: {retrieve_response.status_code}")
                if retrieve_response.status_code == 200:
                    memories = retrieve_response.json()
                    print(f"Memories retrieved: {len(memories.get('memories', []))}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"✗ Memory Test Failed: {e}")
            return False
    
    def test_agent_orchestration(self, task: str = "Create a simple Python function to calculate fibonacci numbers"):
        """Test the agent orchestration endpoint"""
        try:
            payload = {
                "task": task,
                "agent_type": "coding",
                "priority": "medium"
            }
            
            response = self.session.post(
                f"{self.backend_url}/api/agents/orchestrate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"\n=== Agent Orchestration Test ===")
            print(f"Task: {task}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Agent assigned: {result.get('agent_type', 'N/A')}")
                print(f"Task ID: {result.get('task_id', 'N/A')}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Agent Orchestration Test Failed: {e}")
            return False
    
    def run_comprehensive_test_suite(self):
        """Run all available tests"""
        print("\n" + "="*60)
        print("SENTIENT CORE - COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        print(f"\nTesting Services:")
        print(f"- Backend: {self.backend_url}")
        print(f"- Frontend: {self.frontend_url}")
        
        # Health checks
        print("\n--- Health Checks ---")
        backend_healthy = self.test_backend_health()
        frontend_healthy = self.test_frontend_health()
        
        if not backend_healthy:
            print("\n⚠️  Backend not accessible. Skipping API tests.")
            return
        
        # API endpoint tests
        print("\n--- API Endpoint Tests ---")
        
        # Basic chat functionality
        self.test_chat_endpoint("Hello, can you help me understand how this system works?")
        time.sleep(1)
        
        # Follow-up conversation
        self.test_chat_endpoint("What programming languages do you support?")
        time.sleep(1)
        
        # Research functionality
        self.test_research_endpoint("Latest developments in large language models")
        time.sleep(1)
        
        # Memory management
        self.test_memory_endpoint()
        time.sleep(1)
        
        # Agent orchestration
        self.test_agent_orchestration("Design a REST API for a todo application")
        
        print("\n" + "="*60)
        print("TEST SUITE COMPLETED")
        print("="*60)

def print_available_endpoints():
    """Print all available endpoints for manual testing"""
    print("\n" + "="*60)
    print("AVAILABLE ENDPOINTS FOR MANUAL TESTING")
    print("="*60)
    
    endpoints = {
        "Backend API (FastAPI)": {
            "Base URL": "http://localhost:8000",
            "Endpoints": {
                "GET /health": "Health check",
                "GET /docs": "Interactive API documentation (Swagger UI)",
                "GET /redoc": "Alternative API documentation",
                "POST /api/chat": "Main chat interface",
                "POST /api/research": "Research agent endpoint",
                "POST /api/memory/store": "Store conversation memory",
                "GET /api/memory/retrieve/{conversation_id}": "Retrieve memories",
                "POST /api/agents/orchestrate": "Agent task orchestration",
                "GET /api/conversations/{conversation_id}": "Get conversation history",
                "POST /api/upload": "File upload endpoint",
                "GET /api/status": "System status"
            }
        },
        "Frontend (Next.js)": {
            "Base URL": "http://localhost:3000",
            "Pages": {
                "/": "Main chat interface",
                "/research": "Research results page",
                "/memory": "Memory management interface",
                "/agents": "Agent management dashboard",
                "/settings": "System settings"
            }
        }
    }
    
    for service, details in endpoints.items():
        print(f"\n{service}:")
        print(f"  Base URL: {details['Base URL']}")
        
        endpoints_or_pages = details.get('Endpoints', details.get('Pages', {}))
        for endpoint, description in endpoints_or_pages.items():
            print(f"  {endpoint:<40} - {description}")

def print_test_payloads():
    """Print example payloads for testing"""
    print("\n" + "="*60)
    print("EXAMPLE TEST PAYLOADS")
    print("="*60)
    
    payloads = {
        "Chat Request": {
            "endpoint": "POST /api/chat",
            "payload": {
                "message": "Hello, I need help with Python development",
                "conversation_id": "conv-123",
                "user_id": "user-456"
            }
        },
        "Research Request": {
            "endpoint": "POST /api/research",
            "payload": {
                "query": "Best practices for FastAPI development",
                "research_type": "technical",
                "max_results": 10
            }
        },
        "Memory Store": {
            "endpoint": "POST /api/memory/store",
            "payload": {
                "conversation_id": "conv-123",
                "content": "User prefers TypeScript for frontend",
                "memory_type": "preference"
            }
        },
        "Agent Orchestration": {
            "endpoint": "POST /api/agents/orchestrate",
            "payload": {
                "task": "Create a React component for user authentication",
                "agent_type": "frontend",
                "priority": "high",
                "requirements": ["TypeScript", "React Hooks", "Form validation"]
            }
        }
    }
    
    for test_name, details in payloads.items():
        print(f"\n{test_name}:")
        print(f"  Endpoint: {details['endpoint']}")
        print(f"  Payload:")
        print(f"  {json.dumps(details['payload'], indent=4)}")

if __name__ == "__main__":
    # Print available endpoints
    print_available_endpoints()
    
    # Print test payloads
    print_test_payloads()
    
    # Run comprehensive test suite
    tester = SentientCoreTestSuite()
    tester.run_comprehensive_test_suite()
    
    print("\n" + "="*60)
    print("QUICK START TESTING COMMANDS")
    print("="*60)
    print("\n1. Test Backend Health:")
    print("   curl http://localhost:8000/health")
    
    print("\n2. View API Documentation:")
    print("   Open: http://localhost:8000/docs")
    
    print("\n3. Test Chat Endpoint:")
    print('   curl -X POST http://localhost:8000/api/chat \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"message": "Hello, how can you help me?", "conversation_id": "test-001", "user_id": "test-user"}\'')
    
    print("\n4. Access Frontend:")
    print("   Open: http://localhost:3000")
    
    print("\n5. Run this test script:")
    print("   python test_endpoints_guide.py")
    
    print("\n" + "="*60)