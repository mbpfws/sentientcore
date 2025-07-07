#!/usr/bin/env python3
"""Persistent server that tries to stay alive and log everything."""

import socket
import threading
import time
import json
from datetime import datetime
import sys
import traceback

class PersistentServer:
    def __init__(self, host='127.0.0.1', port=8011):
        self.host = host
        self.port = port
        self.running = False
        self.socket = None
        
    def log(self, message):
        print(f"[{datetime.now()}] {message}")
        
    def handle_client(self, client_socket, address):
        """Handle individual client connections."""
        try:
            self.log(f"Client connected from {address}")
            
            # Read the request
            request = client_socket.recv(1024).decode('utf-8')
            self.log(f"Received request: {request[:100]}...")
            
            # Parse the request to get the path
            lines = request.split('\n')
            if lines:
                request_line = lines[0]
                parts = request_line.split(' ')
                if len(parts) >= 2:
                    method = parts[0]
                    path = parts[1]
                    self.log(f"Method: {method}, Path: {path}")
                    
                    # Generate response based on path
                    if path == '/health':
                        response_data = {
                            "status": "healthy",
                            "port": self.port,
                            "timestamp": datetime.now().isoformat(),
                            "server": "persistent_server"
                        }
                    elif path == '/':
                        response_data = {
                            "message": "Persistent server running",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        response_data = {"error": "Not found"}
                    
                    # Create HTTP response
                    response_json = json.dumps(response_data)
                    http_response = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: application/json\r\n"
                        "Access-Control-Allow-Origin: *\r\n"
                        f"Content-Length: {len(response_json)}\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                        f"{response_json}"
                    )
                    
                    client_socket.send(http_response.encode('utf-8'))
                    self.log(f"Response sent to {address}")
                    
        except Exception as e:
            self.log(f"Error handling client {address}: {e}")
            self.log(f"Traceback: {traceback.format_exc()}")
        finally:
            try:
                client_socket.close()
                self.log(f"Client {address} disconnected")
            except:
                pass
    
    def start(self):
        """Start the server."""
        try:
            self.log(f"Starting persistent server on {self.host}:{self.port}")
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind and listen
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            
            self.log(f"Server listening on {self.host}:{self.port}")
            self.log(f"Server socket: {self.socket.getsockname()}")
            
            # Accept connections
            while self.running:
                try:
                    self.log("Waiting for connections...")
                    client_socket, address = self.socket.accept()
                    self.log(f"Connection accepted from {address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        self.log(f"Error accepting connection: {e}")
                        self.log(f"Traceback: {traceback.format_exc()}")
                        time.sleep(1)  # Brief pause before retrying
                    
        except Exception as e:
            self.log(f"Server startup error: {e}")
            self.log(f"Traceback: {traceback.format_exc()}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server."""
        self.log("Stopping server...")
        self.running = False
        if self.socket:
            try:
                self.socket.close()
                self.log("Server socket closed")
            except Exception as e:
                self.log(f"Error closing socket: {e}")

if __name__ == "__main__":
    server = PersistentServer()
    
    try:
        server.log(f"Python version: {sys.version}")
        server.log(f"Platform: {sys.platform}")
        server.start()
    except KeyboardInterrupt:
        server.log("Server interrupted by user")
    except Exception as e:
        server.log(f"Unexpected error: {e}")
        server.log(f"Traceback: {traceback.format_exc()}")
    finally:
        server.stop()
        server.log("Server shutdown complete")