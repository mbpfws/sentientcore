#!/usr/bin/env python3
"""
Real Functionality Test - Demonstrates actual working system capabilities
This test performs real end-to-end operations with actual API calls and agent interactions.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List

# Import the actual system components
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.agents.research_agent import ResearchAgent
from core.models import AppState, Message, ResearchState, ResearchStep, LogEntry
from core.services.llm_service import EnhancedLLMService
from graphs.research_graph import research_app

class RealFunctionalityTester:
    """Tests actual system functionality with real API calls and agent interactions."""
    
    def __init__(self):
        self.results = []
        self.llm_service = None
        
    def log_result(self, test_name: str, status: str, message: str, details: Dict = None):
        """Log test results with timestamp."""
        result = {
            "test_name": test_name,
            "status": status,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        # Console output with emoji
        emoji = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
        print(f"{emoji} {test_name}: {message}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
    
    async def test_llm_service_initialization(self):
        """Test LLM service initialization and basic functionality."""
        print("\n=== Testing LLM Service Initialization ===")
        
        try:
            # Check for API keys
            api_keys_available = {
                "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
                "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
                "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            }
            
            available_keys = [k for k, v in api_keys_available.items() if v]
            
            if not available_keys:
                self.log_result(
                    "LLM Service: API Keys",
                    "FAIL",
                    "No API keys found in environment variables",
                    api_keys_available
                )
                return
            
            self.log_result(
                "LLM Service: API Keys",
                "PASS",
                f"Found {len(available_keys)} API key(s)",
                {"available_keys": available_keys}
            )
            
            # Initialize LLM service
            self.llm_service = EnhancedLLMService()
            
            self.log_result(
                "LLM Service: Initialization",
                "PASS",
                f"Successfully initialized with {len(self.llm_service.providers)} provider(s)",
                {
                    "providers": list(self.llm_service.providers.keys()),
                    "fallback_chain": self.llm_service.fallback_chain
                }
            )
            
        except Exception as e:
            self.log_result(
                "LLM Service: Initialization",
                "FAIL",
                f"Failed to initialize: {str(e)}"
            )
    
    async def test_simple_llm_call(self):
        """Test a simple LLM API call."""
        print("\n=== Testing Simple LLM Call ===")
        
        if not self.llm_service:
            self.log_result(
                "LLM Call: Simple Test",
                "FAIL",
                "LLM service not initialized"
            )
            return
        
        try:
            # Test a simple call
            system_prompt = "You are a helpful assistant. Respond concisely."
            user_prompt = "What is 2+2? Answer in one sentence."
            model = "gpt-4o-mini"  # Use a common model
            
            response = await self.llm_service.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model
            )
            
            if response and isinstance(response, str) and len(response) > 0:
                self.log_result(
                    "LLM Call: Simple Test",
                    "PASS",
                    "Successfully received response from LLM",
                    {
                        "response_length": len(response),
                        "response_preview": response[:100] + "..." if len(response) > 100 else response
                    }
                )
            else:
                self.log_result(
                    "LLM Call: Simple Test",
                    "FAIL",
                    f"Invalid response format: {type(response)}"
                )
                
        except Exception as e:
            self.log_result(
                "LLM Call: Simple Test",
                "FAIL",
                f"LLM call failed: {str(e)}"
            )
    
    async def test_orchestrator_real_conversation(self):
        """Test UltraOrchestrator with real conversation flow."""
        print("\n=== Testing Real Orchestrator Conversation ===")
        
        if not self.llm_service:
            self.log_result(
                "Orchestrator: Real Conversation",
                "FAIL",
                "LLM service not initialized"
            )
            return
        
        try:
            orchestrator = UltraOrchestrator(self.llm_service)
            
            # Create realistic conversation scenario
            app_state = AppState(
                messages=[
                    Message(sender="user", content="I want to build a simple todo app with React and Node.js")
                ],
                tasks=[],
                logs=[]
            )
            
            # Process the conversation
            result = await orchestrator.invoke(app_state)
            
            if isinstance(result, dict) and "messages" in result:
                messages = result.get("messages", [])
                tasks = result.get("tasks", [])
                next_action = result.get("next_action")
                
                self.log_result(
                    "Orchestrator: Real Conversation",
                    "PASS",
                    "Successfully processed user request and generated response",
                    {
                        "messages_count": len(messages),
                        "tasks_count": len(tasks),
                        "next_action": next_action,
                        "response_preview": messages[-1].get("content", "")[:150] + "..." if messages else "No response"
                    }
                )
            else:
                self.log_result(
                    "Orchestrator: Real Conversation",
                    "FAIL",
                    f"Unexpected result format: {type(result)}"
                )
                
        except Exception as e:
            self.log_result(
                "Orchestrator: Real Conversation",
                "FAIL",
                f"Orchestrator test failed: {str(e)}"
            )
    
    async def test_research_agent_real_research(self):
        """Test ResearchAgent with real research functionality."""
        print("\n=== Testing Real Research Agent ===")
        
        if not self.llm_service:
            self.log_result(
                "Research Agent: Real Research",
                "FAIL",
                "LLM service not initialized"
            )
            return
        
        try:
            research_agent = ResearchAgent(self.llm_service)
            
            # Create a realistic research query
            research_state = ResearchState(
                original_query="Best practices for React state management in 2024",
                steps=[],
                logs=[],
                final_report=None
            )
            
            # Test planning
            planned_state = await research_agent.plan_steps(research_state)
            
            if planned_state.steps and len(planned_state.steps) > 0:
                self.log_result(
                    "Research Agent: Planning",
                    "PASS",
                    f"Successfully planned {len(planned_state.steps)} research steps",
                    {
                        "steps_count": len(planned_state.steps),
                        "first_step": planned_state.steps[0].query if planned_state.steps else "None"
                    }
                )
                
                # Test execution of first step only (to avoid long execution)
                if planned_state.steps:
                    executed_state = await research_agent.execute_search(planned_state)
                    
                    completed_steps = [step for step in executed_state.steps if step.status == "completed"]
                    if completed_steps:
                        self.log_result(
                            "Research Agent: Execution",
                            "PASS",
                            f"Successfully executed research step",
                            {
                                "completed_steps": len(completed_steps),
                                "result_preview": completed_steps[0].result[:200] + "..." if completed_steps[0].result else "No result"
                            }
                        )
                    else:
                        self.log_result(
                            "Research Agent: Execution",
                            "PARTIAL",
                            "Research step was initiated but not completed"
                        )
            else:
                self.log_result(
                    "Research Agent: Planning",
                    "FAIL",
                    "No research steps were planned"
                )
                
        except Exception as e:
            self.log_result(
                "Research Agent: Real Research",
                "FAIL",
                f"Research agent test failed: {str(e)}"
            )
    
    async def test_research_workflow_real_execution(self):
        """Test the research workflow with real execution."""
        print("\n=== Testing Real Research Workflow ===")
        
        try:
            # Create a focused research query
            initial_state = ResearchState(
                original_query="TypeScript vs JavaScript for large applications",
                steps=[],
                logs=[],
                final_report=None
            )
            
            # Execute the workflow
            result = await research_app.ainvoke(initial_state)
            
            if result and hasattr(result, 'steps'):
                completed_steps = [step for step in result.steps if step.status == "completed"]
                
                self.log_result(
                    "Research Workflow: Real Execution",
                    "PASS" if completed_steps else "PARTIAL",
                    f"Workflow executed with {len(completed_steps)} completed steps out of {len(result.steps)} total",
                    {
                        "total_steps": len(result.steps),
                        "completed_steps": len(completed_steps),
                        "has_final_report": bool(result.final_report),
                        "report_preview": result.final_report[:200] + "..." if result.final_report else "No report"
                    }
                )
            else:
                self.log_result(
                    "Research Workflow: Real Execution",
                    "FAIL",
                    "Workflow execution failed or returned invalid result"
                )
                
        except Exception as e:
            self.log_result(
                "Research Workflow: Real Execution",
                "FAIL",
                f"Workflow test failed: {str(e)}"
            )
    
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with memory and context."""
        print("\n=== Testing Multi-Turn Conversation ===")
        
        if not self.llm_service:
            self.log_result(
                "Multi-Turn: Conversation",
                "FAIL",
                "LLM service not initialized"
            )
            return
        
        try:
            orchestrator = UltraOrchestrator(self.llm_service)
            
            # Simulate a realistic multi-turn conversation
            conversation_turns = [
                "Hi, I need help with a web development project",
                "I want to build an e-commerce platform with user authentication",
                "It should support product catalog, shopping cart, and payment processing",
                "What technology stack would you recommend?"
            ]
            
            app_state = AppState(messages=[], tasks=[], logs=[])
            
            for i, user_input in enumerate(conversation_turns):
                # Add user message
                app_state.messages.append(Message(sender="user", content=user_input))
                
                # Get orchestrator response
                result_dict = await orchestrator.invoke(app_state)
                
                # Update app_state from result
                if isinstance(result_dict, dict):
                    app_state.messages = [Message(**msg) if isinstance(msg, dict) else msg for msg in result_dict.get("messages", [])]
                    app_state.tasks = result_dict.get("tasks", [])
                    app_state.logs = result_dict.get("logs", [])
                    app_state.next_action = result_dict.get("next_action")
                
                print(f"Turn {i+1}: User: {user_input}")
                if app_state.messages and len(app_state.messages) > i+1:
                    assistant_response = app_state.messages[-1].content
                    print(f"Turn {i+1}: Assistant: {assistant_response[:100]}...")
            
            # Evaluate conversation quality
            if len(app_state.messages) >= len(conversation_turns) * 2:  # User + Assistant messages
                self.log_result(
                    "Multi-Turn: Conversation",
                    "PASS",
                    f"Successfully completed {len(conversation_turns)}-turn conversation",
                    {
                        "conversation_turns": len(conversation_turns),
                        "total_messages": len(app_state.messages),
                        "final_action": app_state.next_action,
                        "tasks_generated": len(app_state.tasks)
                    }
                )
            else:
                self.log_result(
                    "Multi-Turn: Conversation",
                    "PARTIAL",
                    "Conversation partially completed but some responses missing"
                )
                
        except Exception as e:
            self.log_result(
                "Multi-Turn: Conversation",
                "FAIL",
                f"Multi-turn conversation failed: {str(e)}"
            )
    
    async def run_all_tests(self):
        """Run all real functionality tests."""
        print("🚀 Starting Real Functionality Tests...\n")
        print("This test suite performs actual API calls and agent interactions.")
        print("Please ensure you have valid API keys configured.\n")
        
        # Run all tests
        await self.test_llm_service_initialization()
        await self.test_simple_llm_call()
        await self.test_orchestrator_real_conversation()
        await self.test_research_agent_real_research()
        await self.test_research_workflow_real_execution()
        await self.test_multi_turn_conversation()
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        partial_tests = len([r for r in self.results if r["status"] == "PARTIAL"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])
        
        print(f"\n📊 Real Functionality Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ⚠️ Partial: {partial_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Save detailed results
        with open("real_functionality_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "partial": partial_tests,
                    "failed": failed_tests,
                    "success_rate": (passed_tests/total_tests)*100
                },
                "detailed_results": self.results,
                "llm_service_stats": self.llm_service.get_usage_statistics() if self.llm_service else None
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: real_functionality_results.json")
        
        if self.llm_service:
            stats = self.llm_service.get_usage_statistics()
            print(f"\n📈 LLM Service Usage Statistics:")
            print(f"   Total Requests: {stats['total_requests']}")
            print(f"   Total Errors: {stats['total_errors']}")
            print(f"   Providers Used: {list(stats['usage_by_provider'].keys())}")
        
        print(f"\n🎉 Real functionality testing completed!")
        
        # Return success status for automation
        return passed_tests > 0 and failed_tests == 0

if __name__ == "__main__":
    async def main():
        tester = RealFunctionalityTester()
        success = await tester.run_all_tests()
        return 0 if success else 1
    
    exit_code = asyncio.run(main())
    exit(exit_code)