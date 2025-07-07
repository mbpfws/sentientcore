#!/usr/bin/env python3
"""
Direct socket client test to bypass HTTP libraries
"""

import socket
import time
import sys

def test_socket_connection(host, port):
    """Test direct socket connection"""
    try:
        print(f"Attempting to connect to {host}:{port}...")
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        
        # Try to connect
        result = sock.connect_ex((host, port))
        
        if result == 0:
            print(f"✅ Successfully connected to {host}:{port}")
            
            # Try to send a simple HTTP request
            request = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n"
            sock.send(request.encode())
            
            # Try to receive response
            response = sock.recv(1024).decode()
            print(f"Response received: {response[:200]}...")
            
            sock.close()
            return True
        else:
            print(f"❌ Failed to connect to {host}:{port} - Error code: {result}")
            sock.close()
            return False
            
    except Exception as e:
        print(f"❌ Socket connection error: {e}")
        return False

def test_localhost_resolution():
    """Test if localhost resolves correctly"""
    try:
        print("Testing localhost resolution...")
        ip = socket.gethostbyname('localhost')
        print(f"✅ localhost resolves to: {ip}")
        return ip
    except Exception as e:
        print(f"❌ localhost resolution failed: {e}")
        return None

def test_loopback_interface():
    """Test if we can connect to 127.0.0.1 directly"""
    try:
        print("Testing loopback interface (127.0.0.1)...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        
        # Test if we can at least create a connection attempt
        result = sock.connect_ex(('127.0.0.1', 80))  # Try port 80
        print(f"Connection attempt to 127.0.0.1:80 result: {result}")
        
        sock.close()
        return True
    except Exception as e:
        print(f"❌ Loopback test failed: {e}")
        return False

def start_simple_echo_server():
    """Start a very simple echo server for testing"""
    try:
        print("Starting simple echo server on port 8004...")
        
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('localhost', 8004))
        server_sock.listen(1)
        
        print("✅ Echo server started, listening...")
        
        # Accept one connection and echo back
        server_sock.settimeout(10)  # 10 second timeout
        client_sock, addr = server_sock.accept()
        print(f"✅ Connection accepted from {addr}")
        
        data = client_sock.recv(1024)
        print(f"Received: {data.decode()[:100]}...")
        
        response = "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nEcho response"
        client_sock.send(response.encode())
        
        client_sock.close()
        server_sock.close()
        
        print("✅ Echo server completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Echo server failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Socket Client Test ===")
    print(f"Python version: {sys.version}")
    
    # Test localhost resolution
    localhost_ip = test_localhost_resolution()
    
    # Test loopback interface
    loopback_ok = test_loopback_interface()
    
    # Test connection to known open port (3000 from earlier test)
    print("\n=== Testing connection to known open port ===")
    port_3000_ok = test_socket_connection('localhost', 3000)
    
    # Test our server ports
    print("\n=== Testing our server ports ===")
    ports_to_test = [8000, 8001, 8002, 8003]
    
    for port in ports_to_test:
        test_socket_connection('localhost', port)
    
    print("\n=== Summary ===")
    print(f"Localhost resolution: {'✅' if localhost_ip else '❌'}")
    print(f"Loopback interface: {'✅' if loopback_ok else '❌'}")
    print(f"Port 3000 connection: {'✅' if port_3000_ok else '❌'}")
    
    if localhost_ip and loopback_ok:
        print("\n✅ Basic networking seems functional")
        if not port_3000_ok:
            print("⚠️  But connections to localhost ports are failing")
            print("This suggests a firewall or security software issue")
    else:
        print("\n❌ Fundamental networking problems detected")