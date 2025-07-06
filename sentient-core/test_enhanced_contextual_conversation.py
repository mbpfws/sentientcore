#!/usr/bin/env python3
"""
Test Enhanced Contextual Conversation

This script tests the improved Ultra Orchestrator's ability to maintain
contextual awareness and build cumulative understanding across conversation turns.
"""

import asyncio
import json
import requests
import uuid
from typing import List, Dict, Any
from datetime import datetime

class EnhancedContextualConversationTest:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"enhanced-context-test-{uuid.uuid4().hex[:8]}"
        self.conversation_log = []
        
    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the orchestrator and return the response."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat/message/json",
                json={
                    "message": message,
                    "session_id": self.session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.conversation_log.append({
                    "turn": len(self.conversation_log) + 1,
                    "user_message": message,
                    "assistant_response": result.get("response", "No response"),
                    "action_type": result.get("action_type", "unknown"),
                    "information_assessment": result.get("information_assessment", "unknown"),
                    "timestamp": datetime.now().isoformat()
                })
                return result
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}", "response": "Error occurred"}
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            return {"error": str(e), "response": "Request failed"}
    
    def analyze_contextual_awareness(self, response: str, turn: int, previous_info: List[str]) -> Dict[str, Any]:
        """Analyze if the response shows proper contextual awareness."""
        response_lower = response.lower()
        
        # Context acknowledgment indicators
        acknowledgment_phrases = [
            "based on our conversation", "building on what you", "from what you", 
            "you mentioned", "so far", "previously", "earlier", "adding to",
            "now that i understand", "perfect!", "excellent!", "great!"
        ]
        
        # Progressive understanding indicators
        progression_phrases = [
            "so far i know", "i now understand", "we have", "complete picture",
            "adding this to", "to complete my understanding"
        ]
        
        # Research readiness indicators
        research_phrases = [
            "sufficient information", "complete picture", "proceed with research",
            "i have enough", "ready to research"
        ]
        
        acknowledges_context = any(phrase in response_lower for phrase in acknowledgment_phrases)
        shows_progression = any(phrase in response_lower for phrase in progression_phrases)
        ready_for_research = any(phrase in response_lower for phrase in research_phrases)
        
        # Check for redundant questions (asking for info already provided)
        redundant_questions = False
        if turn > 1:
            for info in previous_info:
                if info.lower() in response_lower and "?" in response:
                    redundant_questions = True
                    break
        
        return {
            "acknowledges_context": acknowledges_context,
            "shows_progression": shows_progression,
            "ready_for_research": ready_for_research,
            "redundant_questions": redundant_questions,
            "contextual_score": sum([acknowledges_context, shows_progression, not redundant_questions])
        }
    
    def run_enhanced_conversation_test(self):
        """Run the enhanced contextual conversation test."""
        print("🚀 Starting Enhanced Contextual Conversation Test")
        print(f"📋 Session ID: {self.session_id}")
        print("="*60)
        
        # Progressive conversation flow
        conversation_flow = [
            {
                "message": "I need help with a project",
                "expected_info": [],
                "should_ask_for": "project type"
            },
            {
                "message": "It's a web application",
                "expected_info": ["web application"],
                "should_ask_for": "purpose/use case"
            },
            {
                "message": "For managing customer data",
                "expected_info": ["web application", "customer data management"],
                "should_ask_for": "technology stack"
            },
            {
                "message": "Using React and Node.js",
                "expected_info": ["web application", "customer data", "React", "Node.js"],
                "should_ask_for": "database and authentication"
            },
            {
                "message": "PostgreSQL database with JWT authentication",
                "expected_info": ["web application", "customer data", "React", "Node.js", "PostgreSQL", "JWT"],
                "should_trigger": "research"
            }
        ]
        
        contextual_scores = []
        
        for i, turn in enumerate(conversation_flow, 1):
            print(f"\n🔄 Turn {i}: {turn['message']}")
            
            # Send message
            response = self.send_message(turn['message'])
            assistant_response = response.get("response", "No response")
            action_type = response.get("action_type", "unknown")
            information_assessment = response.get("information_assessment", "unknown")
            
            print(f"🤖 ASSISTANT: {assistant_response}")
            print(f"📊 Action Type: {action_type}")
            print(f"📈 Information Assessment: {information_assessment}")
            
            # Analyze contextual awareness (skip for first turn)
            if i > 1:
                analysis = self.analyze_contextual_awareness(
                    assistant_response, i, turn['expected_info']
                )
                contextual_scores.append(analysis['contextual_score'])
                
                print(f"📊 Context Analysis:")
                print(f"  ✓ Acknowledges context: {analysis['acknowledges_context']}")
                print(f"  ✓ Shows progression: {analysis['shows_progression']}")
                print(f"  ✓ Ready for research: {analysis['ready_for_research']}")
                print(f"  ❌ Redundant questions: {analysis['redundant_questions']}")
                print(f"  📈 Contextual Score: {analysis['contextual_score']}/3")
                
                if analysis['contextual_score'] >= 2:
                    print(f"  ✅ GOOD contextual awareness")
                else:
                    print(f"  ❌ POOR contextual awareness")
            
            # Check if final turn should trigger research
            if i == len(conversation_flow):
                if action_type == "research" or "sufficient" in information_assessment.lower():
                    print(f"  ✅ Correctly identified sufficient information for research")
                else:
                    print(f"  ❌ Failed to recognize sufficient information for research")
        
        # Final analysis
        print("\n" + "="*60)
        print(f"📋 Final Analysis:")
        print(f"Total conversation turns: {len(conversation_flow)}")
        print(f"Session ID for manual review: {self.session_id}")
        print(f"View full conversation at: {self.base_url}/api/sessions/{self.session_id}/history")
        
        if contextual_scores:
            avg_score = sum(contextual_scores) / len(contextual_scores)
            print(f"📊 Average Contextual Score: {avg_score:.1f}/3.0")
            
            if avg_score >= 2.0:
                print(f"✅ EXCELLENT contextual awareness - Enhanced prompt working!")
            elif avg_score >= 1.5:
                print(f"⚠️ MODERATE contextual awareness - Some improvement")
            else:
                print(f"❌ POOR contextual awareness - Enhancement needs more work")
        
        # Check final response for research trigger
        final_response = self.conversation_log[-1] if self.conversation_log else None
        if final_response:
            if (final_response['action_type'] == 'research' or 
                'sufficient' in final_response['information_assessment'].lower()):
                print(f"💬 Final response appropriately suggests research")
            else:
                print(f"❌ Final response failed to trigger research despite sufficient info")
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Conversation log contains {len(self.conversation_log)} exchanges")
        
        return {
            "session_id": self.session_id,
            "conversation_log": self.conversation_log,
            "contextual_scores": contextual_scores,
            "average_score": sum(contextual_scores) / len(contextual_scores) if contextual_scores else 0
        }

def main():
    """Run the enhanced contextual conversation test."""
    test = EnhancedContextualConversationTest()
    result = test.run_enhanced_conversation_test()
    
    # Save detailed results
    with open(f"enhanced_context_test_results_{result['session_id']}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: enhanced_context_test_results_{result['session_id']}.json")

if __name__ == "__main__":
    main()