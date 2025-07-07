#!/usr/bin/env python3
"""
Windows Diagnostic Script
Investigates potential Windows-specific issues causing server crashes
"""

import socket
import sys
import os
import subprocess
import platform
import time
from datetime import datetime

def test_socket_binding():
    """Test if we can bind to various ports"""
    print("\n=== Socket Binding Test ===")
    
    ports_to_test = [8000, 8001, 8002, 8003, 9000]
    
    for port in ports_to_test:
        try:
            # Test TCP socket binding
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            sock.listen(1)
            print(f"✓ Port {port}: Successfully bound and listening")
            sock.close()
        except Exception as e:
            print(f"✗ Port {port}: Failed to bind - {e}")

def test_localhost_connectivity():
    """Test basic localhost connectivity"""
    print("\n=== Localhost Connectivity Test ===")
    
    try:
        # Test if we can connect to localhost
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 80))  # Try port 80
        if result == 0:
            print("✓ Can connect to localhost:80")
        else:
            print(f"✗ Cannot connect to localhost:80 - Error code: {result}")
        sock.close()
    except Exception as e:
        print(f"✗ Localhost connectivity test failed: {e}")

def check_firewall_status():
    """Check Windows Firewall status"""
    print("\n=== Windows Firewall Status ===")
    
    try:
        result = subprocess.run(
            ['netsh', 'advfirewall', 'show', 'allprofiles', 'state'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("Firewall Status:")
            print(result.stdout)
        else:
            print(f"Failed to check firewall status: {result.stderr}")
    except Exception as e:
        print(f"Error checking firewall: {e}")

def check_antivirus_processes():
    """Check for common antivirus processes that might interfere"""
    print("\n=== Antivirus Process Check ===")
    
    antivirus_processes = [
        'avp.exe', 'avgnt.exe', 'avguard.exe', 'bdagent.exe',
        'msmpeng.exe', 'windefend.exe', 'mcshield.exe',
        'nortonsecurity.exe', 'avastui.exe', 'avgui.exe'
    ]
    
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        if result.returncode == 0:
            running_processes = result.stdout.lower()
            found_av = []
            for av_proc in antivirus_processes:
                if av_proc.lower() in running_processes:
                    found_av.append(av_proc)
            
            if found_av:
                print(f"Found antivirus processes: {', '.join(found_av)}")
            else:
                print("No common antivirus processes detected")
        else:
            print("Failed to list processes")
    except Exception as e:
        print(f"Error checking processes: {e}")

def test_python_server_minimal():
    """Test the most minimal Python server possible"""
    print("\n=== Minimal Python Server Test ===")
    
    try:
        import threading
        import http.server
        import socketserver
        
        PORT = 8888
        
        class TestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Test server working')
            
            def log_message(self, format, *args):
                print(f"[{datetime.now()}] {format % args}")
        
        print(f"Starting minimal server on port {PORT}...")
        
        with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
            print(f"Server started successfully on port {PORT}")
            
            # Run server in background thread
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            # Wait a moment
            time.sleep(2)
            
            # Test connection
            try:
                import urllib.request
                response = urllib.request.urlopen(f'http://127.0.0.1:{PORT}', timeout=5)
                content = response.read().decode()
                print(f"✓ Server test successful: {content}")
            except Exception as e:
                print(f"✗ Server test failed: {e}")
            
            httpd.shutdown()
            print("Server stopped")
            
    except Exception as e:
        print(f"✗ Minimal server test failed: {e}")
        import traceback
        traceback.print_exc()

def check_system_info():
    """Display system information"""
    print("\n=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Current user: {os.getenv('USERNAME', 'Unknown')}")
    print(f"Working directory: {os.getcwd()}")

def main():
    print("Windows Server Diagnostic Tool")
    print("=" * 50)
    
    check_system_info()
    test_socket_binding()
    test_localhost_connectivity()
    check_firewall_status()
    check_antivirus_processes()
    test_python_server_minimal()
    
    print("\n=== Diagnostic Complete ===")
    print("If all tests pass but FastAPI/uvicorn still fails,")
    print("the issue might be with the specific libraries or event loop.")

if __name__ == "__main__":
    main()