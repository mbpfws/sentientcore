#!/usr/bin/env python3
"""
Comprehensive Diagnostic Report for SentientCore Server Issues

This script diagnoses why the FastAPI servers are failing to start or stay running.
"""

import sys
import os
import subprocess
import socket
import traceback
from datetime import datetime

def check_python_environment():
    """Check Python version and virtual environment"""
    print("=== Python Environment ===")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Virtual Environment: {os.environ.get('VIRTUAL_ENV', 'Not detected')}")
    print(f"Current Working Directory: {os.getcwd()}")
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("=== Dependency Check ===")
    required_packages = ['fastapi', 'uvicorn', 'requests']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} - OK")
        except ImportError as e:
            print(f"✗ {package} - MISSING: {e}")
    print()

def check_port_availability():
    """Check if ports 8000 and 8001 are available"""
    print("=== Port Availability ===")
    ports = [8000, 8001]
    
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            print(f"✓ Port {port} - Available")
            sock.close()
        except OSError as e:
            print(f"✗ Port {port} - {e}")
    print()

def test_basic_fastapi():
    """Test if FastAPI can be imported and run"""
    print("=== FastAPI Basic Test ===")
    try:
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "working"}
        
        print("✓ FastAPI import and app creation - OK")
        
        # Try to get the app configuration
        print(f"✓ App title: {app.title}")
        print(f"✓ App version: {app.version}")
        
    except Exception as e:
        print(f"✗ FastAPI test failed: {e}")
        traceback.print_exc()
    print()

def check_firewall_and_network():
    """Check network and firewall issues"""
    print("=== Network & Firewall Check ===")
    
    # Test localhost resolution
    try:
        import socket
        ip = socket.gethostbyname('localhost')
        print(f"✓ localhost resolves to: {ip}")
    except Exception as e:
        print(f"✗ localhost resolution failed: {e}")
    
    # Test if we can create a simple socket server
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.bind(('127.0.0.1', 0))  # Bind to any available port
        port = test_socket.getsockname()[1]
        test_socket.listen(1)
        print(f"✓ Can create socket server on port {port}")
        test_socket.close()
    except Exception as e:
        print(f"✗ Socket server creation failed: {e}")
    print()

def check_sentientcore_imports():
    """Check if SentientCore modules can be imported"""
    print("=== SentientCore Module Check ===")
    
    # Add the current directory to Python path
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    
    modules_to_test = [
        'app.api.app',
        'core.services.memory_service',
        'core.graphs.sentient_workflow_graph'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module} - OK")
        except Exception as e:
            print(f"✗ {module} - FAILED: {e}")
    print()

def run_minimal_server_test():
    """Try to run a minimal server and capture any errors"""
    print("=== Minimal Server Test ===")
    
    try:
        from fastapi import FastAPI
        import uvicorn
        import threading
        import time
        
        app = FastAPI()
        
        @app.get("/diagnostic")
        def diagnostic():
            return {"message": "Diagnostic server working", "timestamp": datetime.now().isoformat()}
        
        # Try to start server in a separate thread
        def start_server():
            try:
                uvicorn.run(app, host="127.0.0.1", port=9999, log_level="error")
            except Exception as e:
                print(f"Server thread error: {e}")
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Test connection
        import requests
        response = requests.get("http://127.0.0.1:9999/diagnostic", timeout=5)
        print(f"✓ Minimal server test successful: {response.json()}")
        
    except Exception as e:
        print(f"✗ Minimal server test failed: {e}")
        traceback.print_exc()
    print()

def generate_recommendations():
    """Generate recommendations based on findings"""
    print("=== Recommendations ===")
    print("Based on the diagnostic results above:")
    print()
    print("1. If dependencies are missing:")
    print("   pip install fastapi uvicorn requests")
    print()
    print("2. If ports are occupied:")
    print("   Try using different ports (8002, 8003, etc.)")
    print()
    print("3. If firewall is blocking:")
    print("   Check Windows Firewall settings")
    print("   Try running as administrator")
    print()
    print("4. If SentientCore modules fail to import:")
    print("   Check for circular imports or missing dependencies")
    print("   Review the specific error messages above")
    print()
    print("5. If all tests pass but servers still fail:")
    print("   The issue may be in the SentientCore application code")
    print("   Check for initialization errors in the main application")
    print()

def main():
    """Run all diagnostic tests"""
    print("SentientCore Server Diagnostic Report")
    print("=" * 50)
    print(f"Generated at: {datetime.now().isoformat()}")
    print()
    
    check_python_environment()
    check_dependencies()
    check_port_availability()
    test_basic_fastapi()
    check_firewall_and_network()
    check_sentientcore_imports()
    run_minimal_server_test()
    generate_recommendations()
    
    print("=" * 50)
    print("Diagnostic complete. Please review the results above.")

if __name__ == "__main__":
    main()