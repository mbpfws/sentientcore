#!/usr/bin/env python3
"""
Connection Test - Isolate the issue between sockets and HTTP requests
"""

import socket
import threading
import time
import requests
from datetime import datetime

def log(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def simple_http_server(port):
    """Simple HTTP server that properly handles HTTP requests"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))
        sock.listen(5)
        log(f"HTTP server listening on 127.0.0.1:{port}")
        
        while True:
            try:
                client, addr = sock.accept()
                log(f"Connection from {addr}")
                
                # Read the full HTTP request
                request_data = b""
                client.settimeout(5)
                
                while True:
                    try:
                        chunk = client.recv(1024)
                        if not chunk:
                            break
                        request_data += chunk
                        if b"\r\n\r\n" in request_data:
                            break
                    except socket.timeout:
                        break
                
                request_str = request_data.decode('utf-8', errors='ignore')
                log(f"Request received: {len(request_str)} bytes")
                if request_str:
                    first_line = request_str.split('\n')[0]
                    log(f"First line: {first_line}")
                
                # Send proper HTTP response
                response_body = '{"status": "ok", "message": "Hello from simple server", "port": ' + str(port) + '}'
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: " + str(len(response_body)) + "\r\n"
                    "Connection: close\r\n"
                    "\r\n" +
                    response_body
                )
                
                client.send(response.encode('utf-8'))
                client.close()
                log(f"Response sent to {addr}")
                
            except Exception as e:
                log(f"Error handling client: {e}")
                try:
                    client.close()
                except:
                    pass
                
    except Exception as e:
        log(f"Server error: {e}")
    finally:
        try:
            sock.close()
        except:
            pass

def test_socket_connection(port):
    """Test raw socket connection"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', port))
        if result == 0:
            log(f"✓ Raw socket connection to port {port} successful")
            sock.close()
            return True
        else:
            log(f"✗ Raw socket connection to port {port} failed: {result}")
            return False
    except Exception as e:
        log(f"✗ Raw socket connection to port {port} error: {e}")
        return False

def test_http_request(port):
    """Test HTTP request using requests library"""
    try:
        log(f"Attempting HTTP request to 127.0.0.1:{port}")
        response = requests.get(f'http://127.0.0.1:{port}', timeout=10)
        log(f"✓ HTTP request to port {port} successful: {response.status_code}")
        log(f"Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError as e:
        log(f"✗ HTTP request to port {port} failed: ConnectionError - {e}")
        return False
    except requests.exceptions.Timeout as e:
        log(f"✗ HTTP request to port {port} failed: Timeout - {e}")
        return False
    except Exception as e:
        log(f"✗ HTTP request to port {port} failed: {type(e).__name__} - {e}")
        return False

def main():
    log("=== Connection Test Tool ===")
    
    port = 8013
    
    # Start server in background
    log(f"Starting HTTP server on port {port}")
    server_thread = threading.Thread(target=simple_http_server, args=(port,))
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Test raw socket connection
    log("\n=== Testing raw socket connection ===")
    socket_success = test_socket_connection(port)
    
    # Test HTTP request
    log("\n=== Testing HTTP request ===")
    http_success = test_http_request(port)
    
    # Summary
    log("\n=== Summary ===")
    log(f"Raw socket connection: {'✓ SUCCESS' if socket_success else '✗ FAILED'}")
    log(f"HTTP request: {'✓ SUCCESS' if http_success else '✗ FAILED'}")
    
    if socket_success and not http_success:
        log("\n⚠️  ISSUE IDENTIFIED: Raw sockets work but HTTP requests fail!")
        log("This suggests an issue with the requests library, proxy settings, or HTTP protocol handling.")
    elif not socket_success:
        log("\n⚠️  ISSUE IDENTIFIED: Basic socket connections are failing!")
        log("This suggests a fundamental networking or firewall issue.")
    else:
        log("\n✓ Both raw sockets and HTTP requests work correctly.")
    
    # Keep server running for manual testing
    log(f"\nServer will continue running on port {port} for 30 seconds...")
    time.sleep(30)
    log("Test completed.")

if __name__ == "__main__":
    main()