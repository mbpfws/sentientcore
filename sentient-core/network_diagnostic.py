#!/usr/bin/env python3
"""
Network Diagnostic Tool for Sentient Core
Diagnoses server binding and connection issues
"""

import socket
import sys
import time
import threading
from datetime import datetime

def log(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def test_port_binding(port):
    """Test if we can bind to a specific port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))
        sock.listen(1)
        log(f"✓ Successfully bound to port {port}")
        sock.close()
        return True
    except Exception as e:
        log(f"✗ Failed to bind to port {port}: {e}")
        return False

def test_port_connection(port):
    """Test if we can connect to a specific port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if result == 0:
            log(f"✓ Successfully connected to port {port}")
            return True
        else:
            log(f"✗ Failed to connect to port {port}: Connection refused")
            return False
    except Exception as e:
        log(f"✗ Failed to connect to port {port}: {e}")
        return False

def check_listening_ports():
    """Check what ports are actually listening"""
    import subprocess
    try:
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True, shell=True)
        lines = result.stdout.split('\n')
        listening_ports = []
        for line in lines:
            if 'LISTENING' in line and '127.0.0.1:' in line:
                parts = line.split()
                for part in parts:
                    if '127.0.0.1:' in part:
                        port = part.split(':')[-1]
                        listening_ports.append(port)
        log(f"Listening ports on 127.0.0.1: {listening_ports}")
        return listening_ports
    except Exception as e:
        log(f"Failed to check listening ports: {e}")
        return []

def run_simple_server(port):
    """Run a simple HTTP server on specified port"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))
        sock.listen(1)
        log(f"Simple server listening on 127.0.0.1:{port}")
        
        while True:
            try:
                client, addr = sock.accept()
                log(f"Connection from {addr}")
                
                # Read request
                request = client.recv(1024).decode('utf-8')
                log(f"Request: {request.split()[0] if request.split() else 'Empty'}")
                
                # Send response
                response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\": \"ok\", \"port\": " + str(port) + "}"
                client.send(response.encode('utf-8'))
                client.close()
                
            except Exception as e:
                log(f"Error handling client: {e}")
                break
                
    except Exception as e:
        log(f"Failed to start server on port {port}: {e}")
    finally:
        try:
            sock.close()
        except:
            pass

def main():
    log("=== Network Diagnostic Tool ===")
    log(f"Python version: {sys.version}")
    log(f"Platform: {sys.platform}")
    
    # Test ports
    test_ports = [3000, 8000, 8001, 8008, 8009, 8010, 8011]
    
    log("\n=== Checking listening ports ===")
    listening_ports = check_listening_ports()
    
    log("\n=== Testing port binding ===")
    for port in test_ports:
        test_port_binding(port)
    
    log("\n=== Testing port connections ===")
    for port in test_ports:
        test_port_connection(port)
    
    # Start a simple server on port 8012
    log("\n=== Starting simple server on port 8012 ===")
    server_thread = threading.Thread(target=run_simple_server, args=(8012,))
    server_thread.daemon = True
    server_thread.start()
    
    # Wait a bit then test connection
    time.sleep(2)
    log("Testing connection to our simple server...")
    test_port_connection(8012)
    
    # Keep server running for manual testing
    log("\nServer running on port 8012. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("Shutting down...")

if __name__ == "__main__":
    main()