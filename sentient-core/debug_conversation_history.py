#!/usr/bin/env python3
"""
Simple debug script to check conversation_history persistence issue.
"""

import sqlite3
import json
import os

def main():
    db_path = os.path.join(os.getcwd(), "data", "sessions", "sessions.db")
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return
    
    print(f"Database found: {db_path} ({os.path.getsize(db_path)} bytes)")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Get recent sessions
            cursor = conn.execute("""
                SELECT session_id, conversation_history, message_count, last_accessed 
                FROM sessions 
                ORDER BY last_accessed DESC 
                LIMIT 5
            """)
            sessions = cursor.fetchall()
            
            print(f"\nFound {len(sessions)} recent sessions:")
            
            for session_id, conv_history_json, msg_count, last_accessed in sessions:
                print(f"\n--- Session: {session_id} ---")
                print(f"Last accessed: {last_accessed}")
                print(f"Message count: {msg_count}")
                
                if conv_history_json:
                    try:
                        conv_history = json.loads(conv_history_json)
                        print(f"Conversation history entries: {len(conv_history)}")
                        for i, entry in enumerate(conv_history):
                            print(f"  [{i}]: {entry}")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")
                        print(f"Raw data: {conv_history_json}")
                else:
                    print("Conversation history: EMPTY")
                
                # Check actual messages
                msg_cursor = conn.execute("""
                    SELECT sender, content FROM conversation_history 
                    WHERE session_id = ? ORDER BY message_index LIMIT 3
                """, (session_id,))
                messages = msg_cursor.fetchall()
                print(f"Actual messages: {len(messages)}")
                for sender, content in messages:
                    print(f"  {sender}: {content[:100]}...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()