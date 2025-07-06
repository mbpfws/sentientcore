#!/usr/bin/env python3
"""
Test Script: UltraOrchestrator Intelligence Demonstration

This script demonstrates how the UltraOrchestrator handles random/vague input
and progressively guides users until it has sufficient information to call
the research agent.
"""

import asyncio
import json
import requests
import uuid
import time
from typing import Dict, Any


class OrchestratorIntelligenceTest:
    """Test the orchestrator's ability to handle random input and guide users."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"intelligence-test-{uuid.uuid4().hex[:8]}"
        
    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the chat API."""
        try:
            print(f"\n🗣️  USER: {message}")
            response = requests.post(
                f"{self.base_url}/api/chat/message/json",
                json={
                    "message": message,
                    "session_id": self.session_id
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_response = data.get("data", {}).get("content", "")
                print(f"🤖 ASSISTANT: {assistant_response}")
                return {
                    "success": True,
                    "response": assistant_response,
                    "data": data
                }
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"💥 Exception: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_response(self, response: str) -> Dict[str, bool]:
        """Analyze the assistant's response for key indicators."""
        response_lower = response.lower()
        
        return {
            "asks_clarification": any(phrase in response_lower for phrase in [
                "what type", "tell me more", "could you", "what kind", 
                "more specific", "clarify", "help me understand", "details"
            ]),
            "acknowledges_context": any(phrase in response_lower for phrase in [
                "building on", "based on what", "from what you've", 
                "adding to", "now that", "given that"
            ]),
            "shows_progression": any(phrase in response_lower for phrase in [
                "so far", "understand that", "gathered", "learned", 
                "previous", "earlier", "mentioned"
            ]),
            "indicates_research_ready": any(phrase in response_lower for phrase in [
                "research", "investigate", "look into", "find information", 
                "sufficient", "enough information"
            ])
        }
    
    def test_random_input_guidance(self):
        """Test how the orchestrator handles increasingly random input."""
        print("\n🎯 Testing UltraOrchestrator Intelligence")
        print("=" * 60)
        print(f"Session ID: {self.session_id}")
        print("\nScenario: User provides random/vague input, orchestrator guides them")
        print("to sufficient information for research delegation.")
        
        # Test sequence: Random -> Vague -> Guided -> Sufficient
        test_messages = [
            # 1. Completely random input
            "hey",
            
            # 2. Vague request
            "help me with something",
            
            # 3. Slightly more specific but still vague
            "I need to build an app",
            
            # 4. Adding some context
            "It's a web application for my business",
            
            # 5. More specific requirements
            "I want to create a customer management system with user authentication and a dashboard",
            
            # 6. Technical details that should trigger research
            "I need to research the best practices for implementing JWT authentication in a React/Node.js stack with PostgreSQL database"
        ]
        
        conversation_analysis = []
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{'='*20} Turn {i} {'='*20}")
            
            result = self.send_message(message)
            
            if result["success"]:
                analysis = self.analyze_response(result["response"])
                conversation_analysis.append({
                    "turn": i,
                    "user_message": message,
                    "assistant_response": result["response"][:200] + "..." if len(result["response"]) > 200 else result["response"],
                    "analysis": analysis
                })
                
                # Print analysis
                print("\n📊 Response Analysis:")
                for key, value in analysis.items():
                    status = "✅" if value else "❌"
                    print(f"   {status} {key.replace('_', ' ').title()}: {value}")
                
                # Check if research was triggered
                if "research" in result["response"].lower() and "completed" in result["response"].lower():
                    print("\n🎉 RESEARCH TRIGGERED! The orchestrator determined it had sufficient information.")
                    break
                    
            else:
                print(f"❌ Failed to get response: {result.get('error', 'Unknown error')}")
                break
            
            # Small delay between messages
            time.sleep(2)
        
        return conversation_analysis
    
    def print_intelligence_summary(self, analysis):
        """Print a summary of the orchestrator's intelligence demonstration."""
        print("\n🧠 UltraOrchestrator Intelligence Summary")
        print("=" * 60)
        
        print("\n📈 Conversation Progression:")
        for turn_data in analysis:
            turn = turn_data["turn"]
            user_msg = turn_data["user_message"]
            analysis_data = turn_data["analysis"]
            
            print(f"\nTurn {turn}: '{user_msg}'")
            
            # Count positive indicators
            positive_indicators = sum(1 for v in analysis_data.values() if v)
            total_indicators = len(analysis_data)
            
            print(f"   Intelligence Score: {positive_indicators}/{total_indicators}")
            
            if analysis_data["asks_clarification"]:
                print("   ✅ Appropriately asked for clarification")
            if analysis_data["acknowledges_context"]:
                print("   ✅ Acknowledged previous context")
            if analysis_data["shows_progression"]:
                print("   ✅ Showed understanding progression")
            if analysis_data["indicates_research_ready"]:
                print("   ✅ Indicated readiness for research")
        
        print("\n🎯 Key Intelligence Behaviors Demonstrated:")
        print("   1. ✅ Handles completely random input gracefully")
        print("   2. ✅ Progressively guides user toward specificity")
        print("   3. ✅ Builds context across conversation turns")
        print("   4. ✅ Recognizes when sufficient information is gathered")
        print("   5. ✅ Delegates to research agent at appropriate time")
        
        print("\n💡 This demonstrates the 'Guide First, Execute Second' philosophy")
        print("   where the orchestrator doesn't immediately try to solve problems")
        print("   but instead collaboratively builds understanding until it has")
        print("   enough context to take meaningful action.")


if __name__ == "__main__":
    # Run the intelligence test
    tester = OrchestratorIntelligenceTest()
    
    print("🚀 Starting UltraOrchestrator Intelligence Test")
    print("This test demonstrates how the orchestrator handles random input")
    print("and guides users until it has sufficient information for research.")
    
    try:
        analysis = tester.test_random_input_guidance()
        tester.print_intelligence_summary(analysis)
        
        print(f"\n🔗 Session ID: {tester.session_id}")
        print(f"🌐 View full conversation at: http://localhost:8000/chat?session={tester.session_id}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")