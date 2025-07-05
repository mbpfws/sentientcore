#!/usr/bin/env python3
"""
Simple test to verify server connection
"""

import requests
import time
import subprocess
import sys
from threading import Thread

def start_server():
    """Start the minimal server"""
    subprocess.run(["uvicorn", "test_minimal_server:app", "--host", "127.0.0.1", "--port", "8000"])

def test_endpoints():
    """Test server endpoints"""
    # Wait for server to start
    time.sleep(3)
    
    endpoints = [
        "/health",
        "/docs", 
        "/api/state",
        "/api/memory/status",
        "/api/memory/store",
        "/api/agents/execute"
    ]
    
    base_url = "http://127.0.0.1:8000"
    
    print("Testing endpoints...")
    
    for endpoint in endpoints:
        try:
            if endpoint == "/api/memory/store":
                # POST request
                response = requests.post(f"{base_url}{endpoint}", 
                                       json={"content": "test", "metadata": {}}, 
                                       timeout=5)
            elif endpoint == "/api/agents/execute":
                # POST request
                response = requests.post(f"{base_url}{endpoint}", 
                                       json={"agent_type": "monitoring_agent", "task": "test"}, 
                                       timeout=5)
            else:
                # GET request
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            print(f"✅ {endpoint}: {response.status_code} - {response.text[:100]}")
            
        except Exception as e:
            print(f"❌ {endpoint}: Error - {str(e)}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    # Start server in background thread
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Test endpoints
    test_endpoints()