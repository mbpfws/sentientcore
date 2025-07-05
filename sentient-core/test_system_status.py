#!/usr/bin/env python3
"""
SentientCore System Status Dashboard
Quick testing and monitoring script for full-stack system
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

class SystemStatusDashboard:
    def __init__(self):
        self.backend_url = "http://127.0.0.1:8000"
        self.frontend_url = "http://localhost:3000"
        
    def check_backend_health(self) -> Dict[str, Any]:
        """Check backend API health status"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            return {
                "status": "✅ Online",
                "response_time": response.elapsed.total_seconds(),
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "❌ Offline",
                "error": str(e),
                "response_time": None,
                "status_code": None
            }
    
    def check_frontend_health(self) -> Dict[str, Any]:
        """Check frontend server status"""
        try:
            response = requests.get(self.frontend_url, timeout=5)
            return {
                "status": "✅ Online",
                "response_time": response.elapsed.total_seconds(),
                "status_code": response.status_code
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "❌ Offline",
                "error": str(e),
                "response_time": None,
                "status_code": None
            }
    
    def test_api_endpoints(self) -> Dict[str, Any]:
        """Test key API endpoints"""
        endpoints = {
            "docs": "/docs",
            "api_state": "/api/state",
            "api_memory": "/api/memory/status"
        }
        
        results = {}
        for name, endpoint in endpoints.items():
            try:
                response = requests.get(f"{self.backend_url}{endpoint}", timeout=5)
                results[name] = {
                    "status": "✅" if response.status_code == 200 else "⚠️",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except requests.exceptions.RequestException as e:
                results[name] = {
                    "status": "❌",
                    "error": str(e)
                }
        
        return results
    
    def test_memory_storage(self) -> Dict[str, Any]:
        """Test memory storage functionality"""
        test_data = {
            "content": f"Test memory entry - {datetime.now().isoformat()}",
            "metadata": {
                "type": "system_test",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            # Test storage
            store_response = requests.post(
                f"{self.backend_url}/api/memory/store",
                json=test_data,
                timeout=10
            )
            
            if store_response.status_code == 200:
                # Test retrieval
                retrieve_response = requests.get(
                    f"{self.backend_url}/api/memory/retrieve",
                    params={"query": "system_test", "limit": 1},
                    timeout=10
                )
                
                return {
                    "storage": "✅ Success",
                    "retrieval": "✅ Success" if retrieve_response.status_code == 200 else "❌ Failed",
                    "store_response_time": store_response.elapsed.total_seconds(),
                    "retrieve_response_time": retrieve_response.elapsed.total_seconds() if retrieve_response else None
                }
            else:
                return {
                    "storage": "❌ Failed",
                    "error": store_response.text,
                    "status_code": store_response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "storage": "❌ Error",
                "error": str(e)
            }
    
    def test_agent_execution(self) -> Dict[str, Any]:
        """Test basic agent execution"""
        test_request = {
            "agent_type": "monitoring_agent",
            "task": "system_health_check",
            "parameters": {"check_type": "basic"}
        }
        
        try:
            response = requests.post(
                f"{self.backend_url}/api/agents/execute",
                json=test_request,
                timeout=30
            )
            
            return {
                "status": "✅ Success" if response.status_code == 200 else "❌ Failed",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "data": response.json() if response.status_code == 200 else response.text
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "❌ Error",
                "error": str(e)
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive system test"""
        print("🔍 Running SentientCore System Status Check...\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "backend_health": self.check_backend_health(),
            "frontend_health": self.check_frontend_health(),
            "api_endpoints": self.test_api_endpoints(),
            "memory_system": self.test_memory_storage(),
            "agent_system": self.test_agent_execution()
        }
        
        return results
    
    def display_results(self, results: Dict[str, Any]):
        """Display test results in a formatted way"""
        print("="*60)
        print("🚀 SENTIENTCORE SYSTEM STATUS DASHBOARD")
        print("="*60)
        print(f"⏰ Test Time: {results['timestamp']}")
        print()
        
        # Backend Status
        backend = results['backend_health']
        print(f"🔧 BACKEND (FastAPI): {backend['status']}")
        if backend.get('response_time'):
            print(f"   Response Time: {backend['response_time']:.3f}s")
        if backend.get('error'):
            print(f"   Error: {backend['error']}")
        print()
        
        # Frontend Status
        frontend = results['frontend_health']
        print(f"🎨 FRONTEND (Next.js): {frontend['status']}")
        if frontend.get('response_time'):
            print(f"   Response Time: {frontend['response_time']:.3f}s")
        if frontend.get('error'):
            print(f"   Error: {frontend['error']}")
        print()
        
        # API Endpoints
        print("🔗 API ENDPOINTS:")
        for endpoint, data in results['api_endpoints'].items():
            status = data.get('status', '❌')
            response_time = data.get('response_time', 'N/A')
            print(f"   {endpoint}: {status} ({response_time:.3f}s)" if isinstance(response_time, float) else f"   {endpoint}: {status}")
        print()
        
        # Memory System
        memory = results['memory_system']
        print(f"🧠 MEMORY SYSTEM: {memory.get('storage', '❌')}")
        if memory.get('store_response_time'):
            print(f"   Storage Time: {memory['store_response_time']:.3f}s")
        if memory.get('retrieve_response_time'):
            print(f"   Retrieval Time: {memory['retrieve_response_time']:.3f}s")
        print()
        
        # Agent System
        agent = results['agent_system']
        print(f"🤖 AGENT SYSTEM: {agent.get('status', '❌')}")
        if agent.get('response_time'):
            print(f"   Execution Time: {agent['response_time']:.3f}s")
        if agent.get('error'):
            print(f"   Error: {agent['error']}")
        print()
        
        print("="*60)
        print("✅ System check complete!")
        print("📋 See TESTING_GUIDE.md for detailed testing instructions")
        print("="*60)

def main():
    """Main function to run system status check"""
    dashboard = SystemStatusDashboard()
    
    try:
        results = dashboard.run_comprehensive_test()
        dashboard.display_results(results)
        
        # Save results to file
        with open('system_status_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n💾 Results saved to: system_status_results.json")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()