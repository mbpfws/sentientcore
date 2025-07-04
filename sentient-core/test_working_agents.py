#!/usr/bin/env python3
"""
Working Agent Tests - Tests actual agent functionality with proper interfaces
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

# Import the actual agents and models
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.agents.research_agent import ResearchAgent
from core.models import AppState, Message, ResearchState, ResearchStep, LogEntry
from core.services.llm_service import EnhancedLLMService
from graphs.research_graph import research_app

class WorkingAgentTester:
    """Tests actual agent functionality with proper async handling and interfaces."""
    
    def __init__(self):
        self.results = []
        self.llm_service = EnhancedLLMService()
        
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
            print(f"   Details: {details}")
    
    def test_ultra_orchestrator_basic(self):
        """Test UltraOrchestrator basic functionality."""
        print("\n=== Testing UltraOrchestrator Basic Functionality ===")
        
        try:
            # Initialize orchestrator
            orchestrator = UltraOrchestrator(self.llm_service)
            self.log_result(
                "UltraOrchestrator: Initialization",
                "PASS",
                "Successfully initialized UltraOrchestrator"
            )
            
            # Test with a simple app state
            app_state = AppState(
                messages=[
                    Message(sender="user", content="I want to build a simple todo app with user authentication")
                ],
                tasks=[],
                logs=[]
            )
            
            # Test invoke method
            result = orchestrator.invoke(app_state)
            
            if isinstance(result, dict) and "messages" in result:
                self.log_result(
                    "UltraOrchestrator: Invoke Method",
                    "PASS",
                    "Successfully processed user request",
                    {
                        "messages_count": len(result.get("messages", [])),
                        "tasks_count": len(result.get("tasks", [])),
                        "next_action": result.get("next_action")
                    }
                )
            else:
                self.log_result(
                    "UltraOrchestrator: Invoke Method",
                    "FAIL",
                    f"Unexpected result format: {type(result)}"
                )
                
        except Exception as e:
            self.log_result(
                "UltraOrchestrator: Basic Test",
                "FAIL",
                f"Test failed with error: {str(e)}"
            )
    
    def test_research_agent_basic(self):
        """Test ResearchAgent basic functionality."""
        print("\n=== Testing ResearchAgent Basic Functionality ===")
        
        try:
            # Initialize research agent
            research_agent = ResearchAgent(self.llm_service)
            self.log_result(
                "ResearchAgent: Initialization",
                "PASS",
                "Successfully initialized ResearchAgent"
            )
            
            # Test planning functionality
            research_state = ResearchState(
                original_query="Best practices for React authentication",
                steps=[],
                logs=[],
                final_report=None
            )
            
            # Test plan_steps method
            planned_state = research_agent.plan_steps(research_state)
            
            if planned_state.steps and len(planned_state.steps) > 0:
                self.log_result(
                    "ResearchAgent: Planning",
                    "PASS",
                    f"Successfully planned {len(planned_state.steps)} research steps",
                    {
                        "steps_count": len(planned_state.steps),
                        "first_step_query": planned_state.steps[0].query if planned_state.steps else "None"
                    }
                )
            else:
                self.log_result(
                    "ResearchAgent: Planning",
                    "FAIL",
                    "No research steps were planned"
                )
                
        except Exception as e:
            self.log_result(
                "ResearchAgent: Basic Test",
                "FAIL",
                f"Test failed with error: {str(e)}"
            )
    
    async def test_research_workflow_async(self):
        """Test the research workflow graph with proper async handling."""
        print("\n=== Testing Research Workflow (Async) ===")
        
        try:
            # Test workflow import
            self.log_result(
                "Research Workflow: Import",
                "PASS",
                "Successfully imported research workflow"
            )
            
            # Create initial state
            initial_state = ResearchState(
                original_query="Modern JavaScript frameworks comparison 2024",
                steps=[],
                logs=[],
                final_report=None
            )
            
            # Test workflow execution with async
            result = await research_app.ainvoke(initial_state)
            
            if result and hasattr(result, 'steps'):
                completed_steps = [step for step in result.steps if step.status == "completed"]
                self.log_result(
                    "Research Workflow: Execution",
                    "PASS" if completed_steps else "PARTIAL",
                    f"Workflow executed with {len(completed_steps)} completed steps",
                    {
                        "total_steps": len(result.steps),
                        "completed_steps": len(completed_steps),
                        "has_final_report": bool(result.final_report)
                    }
                )
            else:
                self.log_result(
                    "Research Workflow: Execution",
                    "FAIL",
                    "Workflow execution failed or returned invalid result"
                )
                
        except Exception as e:
            self.log_result(
                "Research Workflow: Async Test",
                "FAIL",
                f"Test failed with error: {str(e)}"
            )
    
    def test_conversation_flow(self):
        """Test a realistic conversation flow between user and orchestrator."""
        print("\n=== Testing Realistic Conversation Flow ===")
        
        try:
            orchestrator = UltraOrchestrator(self.llm_service)
            
            # Simulate a multi-turn conversation
            conversations = [
                "Hi, I need help with a project",
                "I want to build a web application for managing personal finances",
                "It should have user accounts, expense tracking, budget planning, and financial reports"
            ]
            
            app_state = AppState(messages=[], tasks=[], logs=[])
            
            for i, user_input in enumerate(conversations):
                # Add user message
                app_state.messages.append(Message(sender="user", content=user_input))
                
                # Get orchestrator response
                result_dict = orchestrator.invoke(app_state)
                
                # Update app_state from result
                app_state.messages = [Message(**msg) for msg in result_dict.get("messages", [])]
                app_state.tasks = result_dict.get("tasks", [])
                app_state.logs = result_dict.get("logs", [])
                app_state.next_action = result_dict.get("next_action")
                
                print(f"Turn {i+1}: User: {user_input}")
                if app_state.messages:
                    last_response = app_state.messages[-1].content
                    print(f"Turn {i+1}: Assistant: {last_response[:100]}...")
            
            # Check final state
            if app_state.next_action == "create_plan" and app_state.tasks:
                self.log_result(
                    "Conversation Flow: Multi-turn",
                    "PASS",
                    f"Successfully guided conversation to plan creation with {len(app_state.tasks)} tasks",
                    {
                        "conversation_turns": len(conversations),
                        "final_action": app_state.next_action,
                        "tasks_created": len(app_state.tasks)
                    }
                )
            else:
                self.log_result(
                    "Conversation Flow: Multi-turn",
                    "PARTIAL",
                    f"Conversation completed but didn't reach plan creation. Final action: {app_state.next_action}"
                )
                
        except Exception as e:
            self.log_result(
                "Conversation Flow: Multi-turn",
                "FAIL",
                f"Test failed with error: {str(e)}"
            )
    
    def test_end_to_end_research(self):
        """Test complete research process from planning to synthesis."""
        print("\n=== Testing End-to-End Research Process ===")
        
        try:
            research_agent = ResearchAgent(self.llm_service)
            
            # Step 1: Plan research
            initial_state = ResearchState(
                original_query="Best database choices for a Node.js e-commerce application",
                steps=[],
                logs=[],
                final_report=None
            )
            
            planned_state = research_agent.plan_steps(initial_state)
            
            if not planned_state.steps:
                self.log_result(
                    "E2E Research: Planning",
                    "FAIL",
                    "No research steps were planned"
                )
                return
            
            # Step 2: Execute one search step (to avoid long execution)
            if planned_state.steps:
                executed_state = research_agent.execute_search(planned_state)
                
                completed_steps = [step for step in executed_state.steps if step.status == "completed"]
                if completed_steps:
                    self.log_result(
                        "E2E Research: Execution",
                        "PASS",
                        f"Successfully executed search step",
                        {
                            "completed_steps": len(completed_steps),
                            "result_length": len(completed_steps[0].result) if completed_steps[0].result else 0
                        }
                    )
                    
                    # Step 3: Synthesize report
                    final_state = research_agent.synthesize_report(executed_state)
                    
                    if final_state.final_report:
                        self.log_result(
                            "E2E Research: Synthesis",
                            "PASS",
                            f"Successfully synthesized final report",
                            {
                                "report_length": len(final_state.final_report),
                                "suggestions_count": len(final_state.continual_search_suggestions or [])
                            }
                        )
                    else:
                        self.log_result(
                            "E2E Research: Synthesis",
                            "FAIL",
                            "Failed to synthesize final report"
                        )
                else:
                    self.log_result(
                        "E2E Research: Execution",
                        "FAIL",
                        "No search steps were completed"
                    )
                    
        except Exception as e:
            self.log_result(
                "E2E Research: Complete Process",
                "FAIL",
                f"Test failed with error: {str(e)}"
            )
    
    async def run_all_tests(self):
        """Run all tests and generate summary."""
        print("🚀 Starting Working Agent Tests...\n")
        
        # Run synchronous tests
        self.test_ultra_orchestrator_basic()
        self.test_research_agent_basic()
        self.test_conversation_flow()
        self.test_end_to_end_research()
        
        # Run async tests
        await self.test_research_workflow_async()
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        partial_tests = len([r for r in self.results if r["status"] == "PARTIAL"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])
        
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ⚠️ Partial: {partial_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Save detailed results
        with open("working_agent_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "partial": partial_tests,
                    "failed": failed_tests,
                    "success_rate": (passed_tests/total_tests)*100
                },
                "detailed_results": self.results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: working_agent_results.json")
        print(f"\n🎉 Working agent testing completed!")

if __name__ == "__main__":
    async def main():
        tester = WorkingAgentTester()
        await tester.run_all_tests()
    
    asyncio.run(main())