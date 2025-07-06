#!/usr/bin/env python3
"""
Test Script: Contextual Conversation Flow
Demonstrates how the UltraOrchestrator should maintain conversation context
and build cumulative understanding across multiple turns.
"""

import requests
import json
import uuid
import time
from typing import Dict, Any, List

class ContextualConversationTest:
    """Test the orchestrator's ability to maintain contextual conversation flow."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"contextual-test-{uuid.uuid4().hex[:8]}"
        self.conversation_log = []
        
    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the chat API and return the response."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat/message/json",
                json={
                    "message": message,
                    "session_id": self.session_id
                },
                timeout=30
            )
            
            result = {
                "status_code": response.status_code,
                "data": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None
            }
            
            # Log the conversation
            self.conversation_log.append({
                "user_message": message,
                "response": result,
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            error_result = {
                "status_code": 500,
                "data": None,
                "error": str(e)
            }
            self.conversation_log.append({
                "user_message": message,
                "response": error_result,
                "timestamp": time.time()
            })
            return error_result
    
    def analyze_contextual_awareness(self, response_content: str, previous_context: List[str]) -> Dict[str, bool]:
        """Analyze if the response shows awareness of previous conversation context."""
        content_lower = response_content.lower()
        
        # Check for contextual references
        contextual_indicators = [
            "building on", "as you mentioned", "from what you've shared", 
            "adding to", "now that i understand", "based on your previous",
            "you've told me", "earlier you said", "continuing from",
            "to summarize what we've discussed", "building upon"
        ]
        
        # Check for acknowledgment of specific previous details
        acknowledges_context = any(indicator in content_lower for indicator in contextual_indicators)
        
        # Check if it references specific details from previous messages
        references_details = False
        for prev_msg in previous_context:
            # Look for key terms from previous messages
            prev_words = set(prev_msg.lower().split())
            response_words = set(content_lower.split())
            # If there's significant overlap, it's likely referencing previous context
            if len(prev_words.intersection(response_words)) > 2:
                references_details = True
                break
        
        # Check if it's asking redundant questions
        asks_redundant = False
        question_indicators = ["what", "how", "when", "where", "why", "which", "?", "tell me"]
        if any(indicator in content_lower for indicator in question_indicators):
            # Check if similar questions were asked before
            for prev_entry in self.conversation_log[:-1]:  # Exclude current response
                if prev_entry["response"]["data"]:
                    prev_response = prev_entry["response"]["data"].get("content", "")
                    if any(indicator in prev_response.lower() for indicator in question_indicators):
                        # Simple check for similar question patterns
                        if len(set(content_lower.split()).intersection(set(prev_response.lower().split()))) > 3:
                            asks_redundant = True
                            break
        
        return {
            "acknowledges_context": acknowledges_context,
            "references_details": references_details,
            "avoids_redundancy": not asks_redundant,
            "shows_progression": acknowledges_context and references_details
        }
    
    def run_contextual_conversation_test(self):
        """Run a test that builds context progressively across multiple turns."""
        print(f"\n🧠 Testing Contextual Conversation Flow")
        print(f"Session ID: {self.session_id}")
        print("=" * 60)
        
        # Progressive conversation that should build context
        conversation_flow = [
            {
                "message": "I need help with a project",
                "expected_behavior": "Should ask for clarification about the project"
            },
            {
                "message": "It's a web application for managing customer data",
                "expected_behavior": "Should acknowledge the web app context and ask for more specific details"
            },
            {
                "message": "I want to use React for the frontend and Node.js for the backend",
                "expected_behavior": "Should reference previous context (web app, customer data) and build on tech stack"
            },
            {
                "message": "The database should be PostgreSQL and I need user authentication",
                "expected_behavior": "Should summarize full context: web app + customer data + React/Node + PostgreSQL + auth"
            },
            {
                "message": "I also need to implement JWT tokens for security",
                "expected_behavior": "Should show complete understanding and potentially suggest research"
            }
        ]
        
        previous_user_messages = []
        
        for i, turn in enumerate(conversation_flow, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {turn['message']}")
            print(f"Expected: {turn['expected_behavior']}")
            
            # Send message
            response = self.send_message(turn["message"])
            
            if response["status_code"] == 200 and response["data"]:
                assistant_response = response["data"].get("content", "")
                print(f"Assistant: {assistant_response[:200]}{'...' if len(assistant_response) > 200 else ''}")
                
                # Analyze contextual awareness
                if i > 1:  # Only analyze from second turn onwards
                    context_analysis = self.analyze_contextual_awareness(assistant_response, previous_user_messages)
                    print(f"\n📊 Context Analysis:")
                    print(f"  ✓ Acknowledges context: {context_analysis['acknowledges_context']}")
                    print(f"  ✓ References details: {context_analysis['references_details']}")
                    print(f"  ✓ Avoids redundancy: {context_analysis['avoids_redundancy']}")
                    print(f"  ✓ Shows progression: {context_analysis['shows_progression']}")
                    
                    # Overall contextual score
                    score = sum(context_analysis.values())
                    print(f"  📈 Contextual Score: {score}/4")
                    
                    if score >= 3:
                        print(f"  🎯 EXCELLENT contextual awareness")
                    elif score >= 2:
                        print(f"  👍 GOOD contextual awareness")
                    else:
                        print(f"  ⚠️  POOR contextual awareness - treating messages independently")
                
            else:
                print(f"❌ Error: {response.get('error', 'Unknown error')}")
            
            previous_user_messages.append(turn["message"])
            time.sleep(1)  # Brief pause between messages
        
        print(f"\n\n📋 Final Analysis:")
        print(f"Total conversation turns: {len(self.conversation_log)}")
        print(f"Session ID for manual review: {self.session_id}")
        print(f"View full conversation at: {self.base_url}/api/sessions/{self.session_id}/history")
        
        # Check if the final response indicates readiness for research
        if self.conversation_log:
            final_response = self.conversation_log[-1]["response"]
            if final_response["data"]:
                final_content = final_response["data"].get("content", "").lower()
                if any(keyword in final_content for keyword in ["research", "investigate", "look into", "find information"]):
                    print(f"🔬 Final response suggests readiness for research delegation")
                else:
                    print(f"💬 Final response continues conversation flow")
        
        return self.conversation_log

def main():
    """Run the contextual conversation test."""
    print("🧪 Contextual Conversation Flow Test")
    print("This test demonstrates how the UltraOrchestrator should:")
    print("1. Build upon previous conversation context")
    print("2. Acknowledge what users have already shared")
    print("3. Avoid asking redundant questions")
    print("4. Show progressive understanding across turns")
    print("5. Determine when sufficient context exists for research")
    
    tester = ContextualConversationTest()
    
    try:
        conversation_log = tester.run_contextual_conversation_test()
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Conversation log contains {len(conversation_log)} exchanges")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()