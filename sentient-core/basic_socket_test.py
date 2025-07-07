#!/usr/bin/env python3
"""
Basic socket test to check if we can bind to ports
"""

import socket
import sys
import time

def test_port_binding(port):
    """Test if we can bind to a specific port"""
    try:
        # Create a socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Try to bind to the port
        sock.bind(('localhost', port))
        sock.listen(1)
        
        print(f"✅ Successfully bound to port {port}")
        
        # Close the socket
        sock.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to bind to port {port}: {e}")
        return False

def test_socket_connection():
    """Test basic socket functionality"""
    try:
        # Test creating a socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("✅ Socket creation successful")
        
        # Test getting local hostname
        hostname = socket.gethostname()
        print(f"✅ Hostname: {hostname}")
        
        # Test getting local IP
        local_ip = socket.gethostbyname(hostname)
        print(f"✅ Local IP: {local_ip}")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"❌ Socket test failed: {e}")
        return False

def check_ports_in_use():
    """Check what ports are currently in use"""
    print("\n=== Checking common ports ===")
    common_ports = [80, 443, 8000, 8001, 8002, 8080, 3000, 5000]
    
    for port in common_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            if result == 0:
                print(f"Port {port}: ✅ OPEN (something is listening)")
            else:
                print(f"Port {port}: ❌ CLOSED")
            sock.close()
        except Exception as e:
            print(f"Port {port}: ❌ ERROR - {e}")

if __name__ == "__main__":
    print("=== Basic Socket and Port Test ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Test basic socket functionality
    print("\n=== Socket Functionality Test ===")
    socket_ok = test_socket_connection()
    
    # Test port binding
    print("\n=== Port Binding Test ===")
    ports_to_test = [8000, 8001, 8002, 8003]
    binding_results = []
    
    for port in ports_to_test:
        result = test_port_binding(port)
        binding_results.append(result)
    
    # Check what ports are in use
    check_ports_in_use()
    
    print("\n=== Summary ===")
    print(f"Socket functionality: {'✅' if socket_ok else '❌'}")
    print(f"Port binding success rate: {sum(binding_results)}/{len(binding_results)}")
    
    if socket_ok and any(binding_results):
        print("\n✅ Basic networking appears to be working")
        print("The issue might be with FastAPI/Uvicorn or HTTP libraries")
    else:
        print("\n❌ Fundamental networking issues detected")
        print("This might be due to:")
        print("  - Firewall blocking connections")
        print("  - Antivirus software interference")
        print("  - Windows network configuration issues")
        print("  - Python installation problems")