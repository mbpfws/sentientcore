#!/usr/bin/env python3
"""
Demonstration of Ultra Orchestrator Contextual Conversation Issue

This script demonstrates the current issue where the orchestrator is not properly
maintaining conversation context and building cumulative understanding across turns.

The user's requirement is that the orchestrator should:
1. Understand the ENTIRE conversation history
2. Connect each turn with previous turns
3. Build cumulative understanding
4. Only execute research when the COMBINATION of all turns provides sufficient information
5. NOT treat each message as unrelated
"""

import requests
import json
import uuid
import time
from typing import Dict, Any, List

class ContextualConversationDemo:
    """Demonstrates the contextual conversation issue and expected behavior."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"context-demo-{uuid.uuid4().hex[:8]}"
        self.conversation_log = []
        
    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the chat API and return the response."""
        try:
            print(f"\n🗣️  USER: {message}")
            response = requests.post(
                f"{self.base_url}/api/chat/message/json",
                json={
                    "message": message,
                    "session_id": self.session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_response = data.get("data", {}).get("content", "")
                print(f"🤖 ASSISTANT: {assistant_response}")
                
                # Log the conversation
                self.conversation_log.append({
                    "user_message": message,
                    "assistant_response": assistant_response,
                    "full_response": data
                })
                
                return {
                    "success": True,
                    "response": assistant_response,
                    "data": data
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def analyze_contextual_behavior(self, response: str, turn_number: int) -> Dict[str, bool]:
        """Analyze if the response shows proper contextual awareness."""
        response_lower = response.lower()
        
        # Check for contextual awareness indicators
        context_indicators = [
            "building on", "based on what you", "from what you've shared",
            "adding to", "now that i understand", "combining", "together with",
            "previously mentioned", "earlier you said", "you also mentioned"
        ]
        
        # Check for progression indicators
        progression_indicators = [
            "so far", "overall", "complete picture", "full understanding",
            "comprehensive", "all the information", "everything you've shared"
        ]
        
        # Check for research readiness based on cumulative info
        research_readiness = [
            "sufficient information", "enough details", "ready to research",
            "comprehensive understanding", "complete requirements"
        ]
        
        acknowledges_context = any(indicator in response_lower for indicator in context_indicators)
        shows_progression = any(indicator in response_lower for indicator in progression_indicators)
        ready_for_research = any(indicator in response_lower for indicator in research_readiness)
        
        return {
            "acknowledges_context": acknowledges_context,
            "shows_progression": shows_progression,
            "ready_for_research": ready_for_research,
            "contextual_score": sum([acknowledges_context, shows_progression, ready_for_research])
        }
    
    def demonstrate_expected_behavior(self):
        """Demonstrate what the expected contextual behavior should look like."""
        print("\n" + "="*80)
        print("DEMONSTRATION: Expected Contextual Conversation Behavior")
        print("="*80)
        
        print("\n📋 SCENARIO: User gradually provides information about a web development project")
        print("🎯 EXPECTED: Orchestrator should build cumulative understanding and only research when sufficient")
        
        # Progressive conversation that should build context
        conversation_turns = [
            {
                "message": "I need help with a project",
                "expected": "Should ask for clarification about project type/domain"
            },
            {
                "message": "It's a web application",
                "expected": "Should acknowledge 'web application' and ask for more specifics"
            },
            {
                "message": "For managing customer data",
                "expected": "Should reference: web app + customer data, ask about tech stack"
            },
            {
                "message": "Using React and Node.js",
                "expected": "Should summarize: web app + customer data + React/Node, ask about database/auth"
            },
            {
                "message": "With PostgreSQL database and JWT authentication",
                "expected": "Should show COMPLETE understanding and suggest research on React/Node/PostgreSQL/JWT best practices"
            }
        ]
        
        print(f"\n🔄 Starting conversation with session: {self.session_id}")
        
        for i, turn in enumerate(conversation_turns, 1):
            print(f"\n--- Turn {i} ---")
            print(f"Expected: {turn['expected']}")
            
            response_data = self.send_message(turn['message'])
            
            if response_data['success']:
                analysis = self.analyze_contextual_behavior(response_data['response'], i)
                
                print(f"\n📊 Context Analysis:")
                print(f"  ✓ Acknowledges context: {analysis['acknowledges_context']}")
                print(f"  ✓ Shows progression: {analysis['shows_progression']}")
                print(f"  ✓ Ready for research: {analysis['ready_for_research']}")
                print(f"  📈 Contextual Score: {analysis['contextual_score']}/3")
                
                if analysis['contextual_score'] >= 2:
                    print(f"  ✅ GOOD contextual awareness")
                elif analysis['contextual_score'] == 1:
                    print(f"  ⚠️  MODERATE contextual awareness")
                else:
                    print(f"  ❌ POOR contextual awareness - treating messages independently")
                
                # Check if research was triggered appropriately
                if i == 5:  # Final turn should trigger research
                    if analysis['ready_for_research']:
                        print(f"  ✅ Correctly identified sufficient information for research")
                    else:
                        print(f"  ❌ Failed to recognize sufficient cumulative information")
                elif i < 5:  # Earlier turns should NOT trigger research
                    if analysis['ready_for_research']:
                        print(f"  ❌ Prematurely triggered research with insufficient context")
                    else:
                        print(f"  ✅ Correctly waiting for more context")
            
            time.sleep(1)  # Brief pause between turns
        
        print(f"\n📋 Final Analysis:")
        print(f"Total conversation turns: {len(self.conversation_log)}")
        print(f"Session ID for manual review: {self.session_id}")
        print(f"View full conversation at: {self.base_url}/api/sessions/{self.session_id}/history")
        
        # Analyze overall conversation flow
        if len(self.conversation_log) >= 5:
            final_response = self.conversation_log[-1]['assistant_response']
            if "research" in final_response.lower():
                print(f"💬 Final response appropriately suggests research")
            else:
                print(f"⚠️  Final response may not have triggered research as expected")
        
        return self.conversation_log
    
    def print_issue_summary(self):
        """Print a summary of the contextual conversation issue."""
        print("\n" + "="*80)
        print("ISSUE SUMMARY: Ultra Orchestrator Contextual Conversation")
        print("="*80)
        
        print("\n🔍 CURRENT PROBLEM:")
        print("   • Orchestrator treats each message independently")
        print("   • Does not build cumulative understanding across turns")
        print("   • Fails to connect previous conversation context")
        print("   • May trigger research prematurely or miss research opportunities")
        
        print("\n✅ EXPECTED BEHAVIOR:")
        print("   • Acknowledge and reference previous conversation turns")
        print("   • Build cumulative understanding progressively")
        print("   • Show progression in understanding (\"Building on what you shared...\")")
        print("   • Only trigger research when COMBINATION of all turns provides sufficient info")
        print("   • Avoid redundant questions already answered in previous turns")
        
        print("\n🎯 USER REQUIREMENT:")
        print("   \"The orchestrator must understand the whole conversation and the next turn")
        print("   must connect with previous turns and in combination if enough to conduct")
        print("   then research is executed not all 6 different messages unrelated like so\"")
        
        print("\n🔧 TECHNICAL IMPLEMENTATION:")
        print("   • Ultra Orchestrator system prompt emphasizes contextual awareness")
        print("   • invoke_state method constructs comprehensive context")
        print("   • Conversation history and cumulative understanding should be maintained")
        print("   • Information assessment should be based on ENTIRE conversation, not just current turn")

def main():
    """Main demonstration function."""
    demo = ContextualConversationDemo()
    
    # Print issue summary first
    demo.print_issue_summary()
    
    # Wait for user to start
    input("\n🚀 Press Enter to start the contextual conversation demonstration...")
    
    # Run the demonstration
    conversation_log = demo.demonstrate_expected_behavior()
    
    print("\n✅ Test completed successfully!")
    print(f"📊 Conversation log contains {len(conversation_log)} exchanges")
    
    return conversation_log

if __name__ == "__main__":
    main()