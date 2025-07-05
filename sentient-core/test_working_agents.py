#!/usr/bin/env python3
"""
Working Agent Tests - Tests actual agent functionality with proper interfaces
"""

import asyncio
import json
from datetime import datetime
from typing import Dict

from core.agents.ultra_orchestrator import UltraOrchestrator
from core.agents.research_agent import ResearchAgent
from core.models import AppState, Message, ResearchState
from core.services.llm_service import EnhancedLLMService
from core.graphs.research_graph import research_app


class WorkingAgentTester:
    def __init__(self):
        self.results = []
        self.llm_service = EnhancedLLMService()

    def log_result(self, test_name: str, status: str, message: str, details: Dict = None):
        result = {
            "test_name": test_name,
            "status": status,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        emoji = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
        print(f"{emoji} {test_name}: {message}")
        if details:
            print(f"   Details: {details}")

    async def test_ultra_orchestrator_basic(self):
        print("\n=== Testing UltraOrchestrator Basic Functionality ===")
        try:
            orchestrator = UltraOrchestrator(self.llm_service)
            self.log_result("UltraOrchestrator: Initialization", "PASS", "Successfully initialized UltraOrchestrator")

            app_state = AppState(
                messages=[Message(sender="user", content="I want to build a simple todo app with user authentication")],
                tasks=[], logs=[]
            )

            result = await orchestrator.invoke(app_state)

            if isinstance(result, AppState):
                self.log_result(
                    "UltraOrchestrator: Invoke Method",
                    "PASS",
                    "Successfully processed user request",
                    {
                        "messages_count": len(result.messages),
                        "tasks_count": len(result.tasks),
                        "next_action": getattr(result, "next_action", None)
                    }
                )
            else:
                self.log_result(
                    "UltraOrchestrator: Invoke Method",
                    "FAIL",
                    f"Unexpected result format: {type(result)}"
                )

        except Exception as e:
            self.log_result("UltraOrchestrator: Basic Test", "FAIL", f"Test failed with error: {str(e)}")

    async def test_research_agent_basic(self):
        print("\n=== Testing ResearchAgent Basic Functionality ===")
        try:
            research_agent = ResearchAgent(self.llm_service)
            self.log_result("ResearchAgent: Initialization", "PASS", "Successfully initialized ResearchAgent")

            research_state = ResearchState(
                original_query="Best practices for React authentication",
                steps=[], logs=[], final_report=None
            )

            planned_state = await research_agent.plan_steps(research_state)

            if planned_state.steps:
                self.log_result(
                    "ResearchAgent: Planning",
                    "PASS",
                    f"Successfully planned {len(planned_state.steps)} research steps",
                    {
                        "steps_count": len(planned_state.steps),
                        "first_step_query": planned_state.steps[0].query
                    }
                )
            else:
                self.log_result("ResearchAgent: Planning", "FAIL", "No research steps were planned")

        except Exception as e:
            self.log_result("ResearchAgent: Basic Test", "FAIL", f"Test failed with error: {str(e)}")

    async def test_conversation_flow(self):
        print("\n=== Testing Realistic Conversation Flow ===")
        try:
            orchestrator = UltraOrchestrator(self.llm_service)

            conversations = [
                "Hi, I need help with a project",
                "I want to build a web application for managing personal finances",
                "It should have user accounts, expense tracking, budget planning, and financial reports"
            ]

            app_state = AppState(messages=[], tasks=[], logs=[])

            for i, user_input in enumerate(conversations):
                app_state.messages.append(Message(sender="user", content=user_input))
                app_state = await orchestrator.invoke(app_state)

                print(f"Turn {i + 1}: User: {user_input}")
                if app_state.messages:
                    print(f"Turn {i + 1}: Assistant: {app_state.messages[-1].content[:100]}...")

            if getattr(app_state, "next_action", None) == "create_plan" and app_state.tasks:
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
            self.log_result("Conversation Flow: Multi-turn", "FAIL", f"Test failed with error: {str(e)}")

    async def test_end_to_end_research(self):
        print("\n=== Testing End-to-End Research Process ===")
        try:
            research_agent = ResearchAgent(self.llm_service)

            initial_state = ResearchState(
                original_query="Best database choices for a Node.js e-commerce application",
                steps=[], logs=[], final_report=None
            )

            planned_state = await research_agent.plan_steps(initial_state)
            if not planned_state.steps:
                self.log_result("E2E Research: Planning", "FAIL", "No research steps were planned")
                return

            executed_state = await research_agent.execute_search(planned_state)
            completed_steps = [step for step in executed_state.steps if step.status == "completed"]

            if completed_steps:
                self.log_result(
                    "E2E Research: Execution",
                    "PASS",
                    "Successfully executed search step",
                    {
                        "completed_steps": len(completed_steps),
                        "result_length": len(completed_steps[0].result) if completed_steps[0].result else 0
                    }
                )

                final_state = await research_agent.synthesize_report(executed_state)

                if final_state.final_report:
                    self.log_result(
                        "E2E Research: Synthesis",
                        "PASS",
                        "Successfully synthesized final report",
                        {
                            "report_length": len(final_state.final_report),
                            "suggestions_count": len(final_state.continual_search_suggestions or [])
                        }
                    )
                else:
                    self.log_result("E2E Research: Synthesis", "FAIL", "Failed to synthesize final report")
            else:
                self.log_result("E2E Research: Execution", "FAIL", "No search steps were completed")

        except Exception as e:
            self.log_result("E2E Research: Complete Process", "FAIL", f"Test failed with error: {str(e)}")

    async def test_research_workflow_async(self):
        print("\n=== Testing Research Workflow (Async) ===")
        try:
            self.log_result("Research Workflow: Import", "PASS", "Successfully imported research workflow")

            initial_state = ResearchState(
                original_query="Modern JavaScript frameworks comparison 2024",
                steps=[], logs=[], final_report=None
            )

            result = await research_app.ainvoke(initial_state)

            if result and hasattr(result, "steps"):
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
                self.log_result("Research Workflow: Execution", "FAIL", "Workflow execution returned invalid result")

        except Exception as e:
            self.log_result("Research Workflow: Async Test", "FAIL", f"Test failed with error: {str(e)}")

    async def run_all_tests(self):
        print("🚀 Starting Working Agent Tests...\n")

        await self.test_ultra_orchestrator_basic()
        await self.test_research_agent_basic()
        await self.test_conversation_flow()
        await self.test_end_to_end_research()
        await self.test_research_workflow_async()

        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        partial_tests = len([r for r in self.results if r["status"] == "PARTIAL"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])

        print("\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ⚠️ Partial: {partial_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

        with open("working_agent_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "partial": partial_tests,
                    "failed": failed_tests,
                    "success_rate": (passed_tests / total_tests) * 100
                },
                "detailed_results": self.results
            }, f, indent=2)

        print("\n💾 Detailed results saved to: working_agent_results.json")
        print("🎉 Working agent testing completed!")


if __name__ == "__main__":
    asyncio.run(WorkingAgentTester().run_all_tests())
