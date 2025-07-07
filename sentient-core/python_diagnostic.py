#!/usr/bin/env python3
"""
Python Environment Diagnostic Script
Diagnoses Python server startup issues
"""

import sys
import os
import subprocess
import socket
import traceback
from pathlib import Path

def check_python_version():
    """Check Python version and environment"""
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Path: {sys.path[:3]}...")  # First 3 entries
    print()

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    venv = os.environ.get('VIRTUAL_ENV')
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    print(f"Virtual Environment: {venv or 'None'}")
    print(f"Conda Environment: {conda_env or 'None'}")
    print()

def check_dependencies():
    """Check critical dependencies"""
    critical_packages = ['fastapi', 'uvicorn', 'requests']
    
    print("Checking Dependencies:")
    for package in critical_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}: Available")
        except ImportError as e:
            print(f"  ✗ {package}: Missing - {e}")
    print()

def check_port_availability():
    """Check if ports are available for binding"""
    test_ports = [8000, 8001, 8002, 8003, 8004, 8005]
    
    print("Port Availability:")
    for port in test_ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                print(f"  ✓ Port {port}: Available")
        except OSError as e:
            print(f"  ✗ Port {port}: {e}")
    print()

def test_basic_server():
    """Test basic HTTP server functionality"""
    print("Testing Basic Server Creation:")
    try:
        import http.server
        import socketserver
        
        # Try to create a basic server without starting it
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(('127.0.0.1', 0), handler) as httpd:
            port = httpd.server_address[1]
            print(f"  ✓ Basic server created successfully on port {port}")
    except Exception as e:
        print(f"  ✗ Basic server creation failed: {e}")
        traceback.print_exc()
    print()

def test_fastapi_import():
    """Test FastAPI import and basic setup"""
    print("Testing FastAPI:")
    try:
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"message": "Hello World"}
        
        print("  ✓ FastAPI app created successfully")
    except Exception as e:
        print(f"  ✗ FastAPI setup failed: {e}")
        traceback.print_exc()
    print()

def test_uvicorn_import():
    """Test Uvicorn import"""
    print("Testing Uvicorn:")
    try:
        import uvicorn
        print(f"  ✓ Uvicorn version: {uvicorn.__version__}")
    except Exception as e:
        print(f"  ✗ Uvicorn import failed: {e}")
        traceback.print_exc()
    print()

def check_firewall_antivirus():
    """Check for potential firewall/antivirus interference"""
    print("System Information:")
    print(f"  OS: {os.name}")
    print(f"  Platform: {sys.platform}")
    
    # Check Windows Defender status (if on Windows)
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['powershell', '-Command', 'Get-MpPreference | Select-Object -Property DisableRealtimeMonitoring'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"  Windows Defender Status: {result.stdout.strip()}")
        except Exception:
            print("  Could not check Windows Defender status")
    print()

def main():
    """Run all diagnostic checks"""
    print("=" * 60)
    print("PYTHON SERVER DIAGNOSTIC REPORT")
    print("=" * 60)
    print()
    
    check_python_version()
    check_virtual_environment()
    check_dependencies()
    check_port_availability()
    test_basic_server()
    test_fastapi_import()
    test_uvicorn_import()
    check_firewall_antivirus()
    
    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()