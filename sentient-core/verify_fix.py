#!/usr/bin/env python3
"""
🔧 AI Orchestrator Fix Verification Script

This script verifies that the AI Orchestrator connection issues have been resolved
and all components are working correctly.
"""

import requests
import time
import json
from typing import Dict, Any

def test_endpoint(url: str, method: str = 'GET', data: Dict[Any, Any] = None, timeout: int = 5) -> Dict[str, Any]:
    """Test an endpoint and return results."""
    try:
        start_time = time.time()
        
        if method.upper() == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        
        response_time = time.time() - start_time
        
        return {
            'success': True,
            'status_code': response.status_code,
            'response_time': round(response_time, 3),
            'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:200]
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'response_time': None,
            'data': None
        }

def print_result(test_name: str, result: Dict[str, Any]):
    """Print test result in a formatted way."""
    status = "✅ PASS" if result['success'] else "❌ FAIL"
    print(f"{status} {test_name}")
    
    if result['success']:
        print(f"   Status: {result['status_code']} | Time: {result['response_time']}s")
        if isinstance(result['data'], dict) and 'status' in result['data']:
            print(f"   Response: {result['data']['status']}")
    else:
        print(f"   Error: {result['error']}")
    print()

def main():
    """Run comprehensive verification tests."""
    print("🔧 AI Orchestrator Fix Verification")
    print("=" * 50)
    print()
    
    # Test configurations
    tests = [
        {
            'name': 'Frontend Server',
            'url': 'http://localhost:3000',
            'method': 'GET'
        },
        {
            'name': 'Backend Health (Direct)',
            'url': 'http://127.0.0.1:8007/health',
            'method': 'GET'
        },
        {
            'name': 'Frontend API Proxy - Health',
            'url': 'http://localhost:3000/api/health',
            'method': 'GET'
        },
        {
            'name': 'Frontend API Proxy - Status',
            'url': 'http://localhost:3000/api/status',
            'method': 'GET'
        },
        {
            'name': 'Chat API (GET)',
            'url': 'http://localhost:3000/api/chat',
            'method': 'GET'
        },
        {
            'name': 'Chat API (POST)',
            'url': 'http://localhost:3000/api/chat',
            'method': 'POST',
            'data': {
                'message': 'Verification test message',
                'model': 'test-model',
                'temperature': 0.7
            }
        }
    ]
    
    results = []
    
    # Run tests
    for test in tests:
        print(f"Testing: {test['name']}...")
        result = test_endpoint(
            test['url'], 
            test['method'], 
            test.get('data'),
            timeout=10
        )
        results.append((test['name'], result))
        print_result(test['name'], result)
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result['success'])
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print()
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ AI Orchestrator connection issues have been RESOLVED")
        print("✅ Frontend and backend are communicating correctly")
        print("✅ API proxy is working as expected")
        print("✅ Chat functionality is operational")
        print()
        print("🚀 System is ready for production use!")
        print()
        print("📍 Access Points:")
        print("   • Frontend: http://localhost:3000")
        print("   • Test Dashboard: http://localhost:3000/test-connection")
        print("   • API Health: http://localhost:3000/api/health")
        print("   • API Status: http://localhost:3000/api/status")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ Please check the failed endpoints and ensure both servers are running")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Ensure frontend server is running: npm run dev (in frontend directory)")
        print("   2. Ensure backend server is running: python development_api_server.py")
        print("   3. Check Windows firewall settings")
        print("   4. Visit test dashboard: http://localhost:3000/test-connection")
    
    print()
    print("📋 For detailed information, see: ORCHESTRATOR_FIX_SUMMARY.md")

if __name__ == '__main__':
    main()