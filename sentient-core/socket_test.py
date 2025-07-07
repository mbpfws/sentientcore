#!/usr/bin/env python3
"""Basic socket test to diagnose networking issues."""

import socket
import sys
from datetime import datetime

def test_socket_binding(port):
    """Test if we can bind to a specific port."""
    print(f"[{datetime.now()}] Testing socket binding on port {port}...")
    
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Try to bind
        sock.bind(('127.0.0.1', port))
        print(f"[{datetime.now()}] Successfully bound to 127.0.0.1:{port}")
        
        # Try to listen
        sock.listen(1)
        print(f"[{datetime.now()}] Successfully listening on port {port}")
        
        # Get actual socket info
        actual_addr = sock.getsockname()
        print(f"[{datetime.now()}] Socket bound to: {actual_addr}")
        
        # Close socket
        sock.close()
        print(f"[{datetime.now()}] Socket closed successfully")
        return True
        
    except Exception as e:
        print(f"[{datetime.now()}] Socket binding failed: {e}")
        print(f"[{datetime.now()}] Error type: {type(e).__name__}")
        try:
            sock.close()
        except:
            pass
        return False

def test_socket_connection(port):
    """Test if we can connect to a port."""
    print(f"[{datetime.now()}] Testing socket connection to port {port}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        
        result = sock.connect_ex(('127.0.0.1', port))
        if result == 0:
            print(f"[{datetime.now()}] Successfully connected to 127.0.0.1:{port}")
            sock.close()
            return True
        else:
            print(f"[{datetime.now()}] Connection failed with error code: {result}")
            sock.close()
            return False
            
    except Exception as e:
        print(f"[{datetime.now()}] Connection test failed: {e}")
        print(f"[{datetime.now()}] Error type: {type(e).__name__}")
        try:
            sock.close()
        except:
            pass
        return False

if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting socket diagnostics...")
    print(f"[{datetime.now()}] Python version: {sys.version}")
    print(f"[{datetime.now()}] Platform: {sys.platform}")
    
    # Test multiple ports
    test_ports = [8007, 8008, 8009, 8010, 8011]
    
    print(f"\n[{datetime.now()}] === BINDING TESTS ===")
    for port in test_ports:
        success = test_socket_binding(port)
        print(f"[{datetime.now()}] Port {port} binding: {'SUCCESS' if success else 'FAILED'}")
        print()
    
    print(f"\n[{datetime.now()}] === CONNECTION TESTS ===")
    for port in test_ports:
        success = test_socket_connection(port)
        print(f"[{datetime.now()}] Port {port} connection: {'SUCCESS' if success else 'FAILED'}")
        print()
    
    print(f"[{datetime.now()}] Socket diagnostics complete")