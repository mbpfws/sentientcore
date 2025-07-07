#!/usr/bin/env python3
"""
Basic Python HTTP server test to isolate networking issues.
"""

import http.server
import socketserver
import threading
import time
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "message": "Basic HTTP server"}')
        else:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "message": "Basic HTTP server is running"}')
    
    def log_message(self, format, *args):
        logger.info(f"HTTP Request: {format % args}")

def start_server():
    """Start the HTTP server in a separate thread"""
    PORT = 8004
    logger.info(f"Starting basic HTTP server on port {PORT}...")
    
    try:
        with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
            logger.info(f"Server running at http://127.0.0.1:{PORT}")
            httpd.serve_forever()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

def test_server():
    """Test the server after a short delay"""
    time.sleep(2)  # Give server time to start
    
    try:
        logger.info("Testing server connection...")
        response = requests.get('http://127.0.0.1:8004/health', timeout=10)
        logger.info(f"Test successful! Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Test the server
    test_server()
    
    # Keep main thread alive
    logger.info("Server is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")