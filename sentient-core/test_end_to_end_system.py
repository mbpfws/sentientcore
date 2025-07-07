#!/usr/bin/env python3
"""
End-to-End System Test
Tests the complete frontend-backend integration to ensure the system works as a whole unit.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EndToEndTester:
    def __init__(self):
        self.backend_url = "http://127.0.0.1:8000"
        self.frontend_url = "http://localhost:3000"
        self.session_id = None
        self.test_results = []
        
    async def test_backend_health(self) -> bool:
        """Test if backend is healthy and responding"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.backend_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Backend health check passed: {data}")
                        return True
                    else:
                        logger.error(f"❌ Backend health check failed with status: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Backend health check failed: {str(e)}")
            return False
    
    async def test_frontend_accessibility(self) -> bool:
        """Test if frontend is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.frontend_url) as response:
                    if response.status == 200:
                        logger.info("✅ Frontend is accessible")
                        return True
                    else:
                        logger.error(f"❌ Frontend accessibility failed with status: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Frontend accessibility failed: {str(e)}")
            return False
    
    async def test_api_endpoints(self) -> bool:
        """Test critical API endpoints"""
        endpoints_to_test = [
            ("/api/status", "GET"),
            ("/api/health", "GET"),
            ("/api/core-services/status", "GET")
        ]
        
        all_passed = True
        
        async with aiohttp.ClientSession() as session:
            for endpoint, method in endpoints_to_test:
                try:
                    url = f"{self.backend_url}{endpoint}"
                    if method == "GET":
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                logger.info(f"✅ {endpoint} endpoint working: {data}")
                            else:
                                logger.error(f"❌ {endpoint} failed with status: {response.status}")
                                all_passed = False
                except Exception as e:
                    logger.error(f"❌ {endpoint} failed: {str(e)}")
                    all_passed = False
        
        return all_passed
    
    async def test_chat_functionality(self) -> bool:
        """Test the complete chat workflow"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test chat message sending
                chat_payload = {
                    "message": "Hello, this is an end-to-end test message",
                    "session_id": f"test_session_{int(time.time())}"
                }
                
                async with session.post(
                    f"{self.backend_url}/api/chat/message",
                    json=chat_payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Chat message sent successfully: {data}")
                        self.session_id = chat_payload["session_id"]
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Chat message failed with status {response.status}: {error_text}")
                        return False
        except Exception as e:
            logger.error(f"❌ Chat functionality test failed: {str(e)}")
            return False
    
    async def test_session_management(self) -> bool:
        """Test session management functionality"""
        if not self.session_id:
            logger.error("❌ No session ID available for testing")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test session retrieval
                async with session.get(f"{self.backend_url}/api/sessions/{self.session_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Session retrieval successful: {data}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Session retrieval failed with status {response.status}: {error_text}")
                        return False
        except Exception as e:
            logger.error(f"❌ Session management test failed: {str(e)}")
            return False
    
    async def test_agent_orchestration(self) -> bool:
        """Test agent orchestration functionality"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test agent status
                async with session.get(f"{self.backend_url}/api/agents/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ Agent orchestration working: {data}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Agent orchestration failed with status {response.status}: {error_text}")
                        return False
        except Exception as e:
            logger.error(f"❌ Agent orchestration test failed: {str(e)}")
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all end-to-end tests"""
        logger.info("🚀 Starting comprehensive end-to-end system test...")
        start_time = datetime.now()
        
        test_suite = [
            ("Backend Health", self.test_backend_health),
            ("Frontend Accessibility", self.test_frontend_accessibility),
            ("API Endpoints", self.test_api_endpoints),
            ("Chat Functionality", self.test_chat_functionality),
            ("Session Management", self.test_session_management),
            ("Agent Orchestration", self.test_agent_orchestration)
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(test_suite)
        
        for test_name, test_func in test_suite:
            logger.info(f"\n🧪 Running test: {test_name}")
            try:
                result = await test_func()
                results[test_name] = {
                    "passed": result,
                    "timestamp": datetime.now().isoformat()
                }
                if result:
                    passed_tests += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "test_results": results
        }
        
        logger.info(f"\n📊 TEST SUMMARY:")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Duration: {duration:.2f} seconds")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! The system is working end-to-end.")
        else:
            logger.error("⚠️  Some tests failed. The system needs attention.")
        
        return summary

async def main():
    """Main test execution"""
    tester = EndToEndTester()
    
    # Run the comprehensive test
    results = await tester.run_comprehensive_test()
    
    # Save results to file
    results_file = f"end_to_end_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📄 Test results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())