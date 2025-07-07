#!/usr/bin/env python3
"""Simple HTTP server using built-in Python modules for testing."""

import http.server
import socketserver
import json
from datetime import datetime
import sys
import socket

class CustomHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"[{datetime.now()}] GET request to {self.path}")
        
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "status": "healthy",
                "port": 8010,
                "timestamp": datetime.now().isoformat(),
                "server": "simple_http_server"
            }
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "message": "Simple HTTP server running",
                "timestamp": datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"error": "Not found"}
            self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[{datetime.now()}] {format % args}")

if __name__ == "__main__":
    PORT = 8010
    
    try:
        print(f"[{datetime.now()}] Testing port {PORT} availability...")
        # Test if port is available
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_socket.bind(('127.0.0.1', PORT))
        test_socket.close()
        print(f"[{datetime.now()}] Port {PORT} is available")
        
        print(f"[{datetime.now()}] Starting simple HTTP server on port {PORT}...")
        print(f"[{datetime.now()}] Python version: {sys.version}")
        print(f"[{datetime.now()}] Platform: {sys.platform}")
        
        with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
            print(f"[{datetime.now()}] Server running at http://127.0.0.1:{PORT}/")
            print(f"[{datetime.now()}] Health endpoint: http://127.0.0.1:{PORT}/health")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] Server interrupted by user")
    except Exception as e:
        print(f"[{datetime.now()}] Server error: {e}")
        import traceback
        print(f"[{datetime.now()}] Traceback: {traceback.format_exc()}")
    finally:
        print(f"[{datetime.now()}] Server shutdown complete")