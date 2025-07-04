#!/usr/bin/env python3
"""
Agent Execution Test
Demonstrates actual agent functionality with real research and orchestration
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class AgentExecutionTester:
    def __init__(self):
        self.results = []
        
    def log_result(self, test_name: str, status: str, details: str = "", data=None):
        """Log test results"""
        result = {
            "test_name": test_name,
            "status": status,
            "details": details,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "ℹ️"
        print(f"{status_symbol} {test_name}: {status}")
        if details:
            print(f"   {details}")
        if data and len(str(data)) < 300:
            print(f"   Data: {data}")
    
    async def test_research_agent_execution(self):
        """Test actual research agent execution"""
        print("\n🔬 Testing Research Agent Execution...")
        
        try:
            from core.agents.research_agent import ResearchAgent
            from core.services.llm_service import EnhancedLLMService
            from core.models import ResearchState, ResearchStep
            
            # Initialize services
            llm_service = EnhancedLLMService()
            research_agent = ResearchAgent(llm_service)
            
            # Create a research state
            research_state = ResearchState(
                original_query="Best Vietnamese language processing libraries for mobile apps",
                steps=[],
                logs=[],
                final_report=None
            )
            
            self.log_result(
                "Research Agent: Initialization",
                "PASS",
                "Successfully initialized ResearchAgent with LLM service"
            )
            
            # Test planning steps
            try:
                planned_state = research_agent.plan_steps(research_state)
                
                if planned_state.steps and len(planned_state.steps) > 0:
                    self.log_result(
                        "Research Agent: Planning",
                        "PASS",
                        f"Generated {len(planned_state.steps)} research steps",
                        {"steps_count": len(planned_state.steps), "first_step": planned_state.steps[0].description[:100] if planned_state.steps else "None"}
                    )
                else:
                    self.log_result(
                        "Research Agent: Planning",
                        "FAIL",
                        "No research steps generated"
                    )
                    
            except Exception as e:
                self.log_result(
                    "Research Agent: Planning",
                    "FAIL",
                    f"Planning failed: {str(e)}"
                )
            
            # Test search execution (with a simple step)
            try:
                # Create a simple research step
                test_step = ResearchStep(
                    query="Vietnamese natural language processing libraries mobile development",
                    status="pending"
                )
                
                search_state = ResearchState(
                    original_query="Vietnamese NLP libraries",
                    steps=[test_step],
                    logs=[],
                    final_report=None
                )
                
                # Execute search
                result_state = research_agent.execute_search(search_state)
                
                if result_state.steps and any(step.result for step in result_state.steps):
                    completed_steps = [step for step in result_state.steps if step.result]
                    self.log_result(
                        "Research Agent: Search Execution",
                        "PASS",
                        f"Completed {len(completed_steps)} research steps",
                        {"completed_steps": len(completed_steps), "first_result": completed_steps[0].result[:100] if completed_steps else "None"}
                    )
                else:
                    self.log_result(
                        "Research Agent: Search Execution",
                        "PARTIAL",
                        "Search executed but no step results generated"
                    )
                    
            except Exception as e:
                self.log_result(
                    "Research Agent: Search Execution",
                    "FAIL",
                    f"Search execution failed: {str(e)}"
                )
                
        except Exception as e:
            self.log_result(
                "Research Agent: Initialization",
                "FAIL",
                f"Failed to initialize: {str(e)}"
            )
    
    async def test_ultra_orchestrator_execution(self):
        """Test actual ultra orchestrator execution"""
        print("\n🎭 Testing Ultra Orchestrator Execution...")
        
        try:
            from core.agents.ultra_orchestrator import UltraOrchestrator
            from core.services.llm_service import EnhancedLLMService
            from core.models import Message, AppState
            
            # Initialize services
            llm_service = EnhancedLLMService()
            orchestrator = UltraOrchestrator(llm_service)
            
            self.log_result(
                "Ultra Orchestrator: Initialization",
                "PASS",
                "Successfully initialized UltraOrchestrator"
            )
            
            # Test message processing
            try:
                test_message = Message(
                    sender="user",
                    content="I need help creating a Vietnamese language learning app with speech recognition"
                )
                
                app_state = AppState(
                    messages=[test_message],
                    current_task=None,
                    context={}
                )
                
                # Process the message
                result = await orchestrator.process_message(app_state)
                
                if result and hasattr(result, 'messages') and len(result.messages) > 1:
                    response_content = result.messages[-1].content
                    self.log_result(
                        "Ultra Orchestrator: Message Processing",
                        "PASS",
                        f"Generated response: {len(response_content)} characters",
                        {"response_length": len(response_content), "response_preview": response_content[:150] + "..." if len(response_content) > 150 else response_content}
                    )
                else:
                    self.log_result(
                        "Ultra Orchestrator: Message Processing",
                        "FAIL",
                        "No response generated or invalid result format"
                    )
                    
            except Exception as e:
                self.log_result(
                    "Ultra Orchestrator: Message Processing",
                    "FAIL",
                    f"Message processing failed: {str(e)}"
                )
                
        except Exception as e:
            self.log_result(
                "Ultra Orchestrator: Initialization",
                "FAIL",
                f"Failed to initialize: {str(e)}"
            )
    
    async def test_workflow_graph_execution(self):
        """Test actual workflow graph execution"""
        print("\n🔄 Testing Workflow Graph Execution...")
        
        try:
            from graphs.research_graph import research_app
            from core.models import ResearchState
            
            self.log_result(
                "Workflow Graph: Import",
                "PASS",
                "Successfully imported research workflow graph"
            )
            
            # Test workflow execution
            try:
                initial_state = ResearchState(
                    original_query="Vietnamese speech recognition APIs for mobile apps",
                    steps=[],
                    logs=[],
                    final_report=None
                )
                
                # Execute the workflow
                result = research_app.invoke(initial_state)
                
                if result and hasattr(result, 'final_report') and result.final_report:
                    self.log_result(
                        "Workflow Graph: Execution",
                        "PASS",
                        f"Workflow completed with report: {len(result.final_report)} characters",
                        {"report_length": len(result.final_report), "steps_executed": len(result.steps), "logs_count": len(result.logs)}
                    )
                elif result and hasattr(result, 'steps') and result.steps:
                    self.log_result(
                        "Workflow Graph: Execution",
                        "PARTIAL",
                        f"Workflow executed with {len(result.steps)} steps but no final report",
                        {"steps_count": len(result.steps), "logs_count": len(result.logs) if hasattr(result, 'logs') else 0}
                    )
                else:
                    self.log_result(
                        "Workflow Graph: Execution",
                        "FAIL",
                        "Workflow execution failed or returned invalid result"
                    )
                    
            except Exception as e:
                self.log_result(
                    "Workflow Graph: Execution",
                    "FAIL",
                    f"Workflow execution failed: {str(e)}"
                )
                
        except Exception as e:
            self.log_result(
                "Workflow Graph: Import",
                "FAIL",
                f"Failed to import workflow: {str(e)}"
            )
    
    async def test_end_to_end_research_workflow(self):
        """Test complete end-to-end research workflow"""
        print("\n🎯 Testing End-to-End Research Workflow...")
        
        try:
            # Import all necessary components
            from core.agents.research_agent import ResearchAgent
            from core.services.llm_service import EnhancedLLMService
            from core.models import ResearchState, Message, AppState
            from core.agents.ultra_orchestrator import UltraOrchestrator
            
            # Initialize services
            llm_service = EnhancedLLMService()
            research_agent = ResearchAgent(llm_service)
            orchestrator = UltraOrchestrator(llm_service)
            
            # Simulate user request
            user_message = Message(
                sender="user",
                content="I need comprehensive research on Vietnamese language processing technologies for building a mobile language learning app"
            )
            
            app_state = AppState(
                messages=[user_message],
                current_task=None,
                context={"research_required": True}
            )
            
            # Step 1: Orchestrator processes the request
            orchestrated_result = await orchestrator.process_message(app_state)
            
            if orchestrated_result:
                self.log_result(
                    "E2E Workflow: Orchestration",
                    "PASS",
                    "Orchestrator successfully processed research request"
                )
                
                # Step 2: Execute research
                research_state = ResearchState(
                    original_query="Vietnamese language processing technologies mobile language learning",
                    steps=[],
                    logs=[],
                    final_report=None
                )
                
                # Plan research steps
                planned_research = research_agent.plan_steps(research_state)
                
                # Execute research
                if planned_research.steps:
                    executed_research = research_agent.execute_search(planned_research)
                    
                    # Generate final report
                    final_research = research_agent.synthesize_report(executed_research)
                    
                    if final_research.final_report:
                        self.log_result(
                            "E2E Workflow: Complete Research",
                            "PASS",
                            f"Complete research workflow executed successfully",
                            {
                                "steps_planned": len(planned_research.steps),
                                "steps_completed": len([s for s in executed_research.steps if s.result]),
                                "report_length": len(final_research.final_report)
                            }
                        )
                    else:
                        self.log_result(
                            "E2E Workflow: Complete Research",
                            "PARTIAL",
                            "Research executed but no final report generated"
                        )
                else:
                    self.log_result(
                        "E2E Workflow: Complete Research",
                        "FAIL",
                        "No research steps planned"
                    )
            else:
                self.log_result(
                    "E2E Workflow: Orchestration",
                    "FAIL",
                    "Orchestrator failed to process request"
                )
                
        except Exception as e:
            self.log_result(
                "E2E Workflow: Complete Research",
                "FAIL",
                f"End-to-end workflow failed: {str(e)}"
            )
    
    def generate_summary(self):
        """Generate test summary"""
        print("\n" + "="*80)
        print("🎯 AGENT EXECUTION TEST REPORT")
        print("="*80)
        
        # Count results by status
        status_counts = {}
        for result in self.results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n📊 Test Summary:")
        for status, count in status_counts.items():
            emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
            print(f"   {emoji} {status}: {count} tests")
        
        print(f"\n📋 Total Tests Run: {len(self.results)}")
        
        # Show working features
        working_features = [r for r in self.results if r["status"] == "PASS"]
        if working_features:
            print("\n✅ Working Agent Features:")
            for feature in working_features:
                print(f"   • {feature['test_name']}")
        
        # Show issues
        issues = [r for r in self.results if r["status"] == "FAIL"]
        if issues:
            print("\n❌ Agent Issues:")
            for issue in issues:
                print(f"   • {issue['test_name']}: {issue['details'][:80]}...")
        
        return {
            "total_tests": len(self.results),
            "status_counts": status_counts,
            "detailed_results": self.results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def run_all_tests(self):
        """Run all agent execution tests"""
        print("🚀 Starting Agent Execution Tests...")
        print("="*80)
        
        # Test individual agents
        await self.test_research_agent_execution()
        await self.test_ultra_orchestrator_execution()
        
        # Test workflow graphs
        await self.test_workflow_graph_execution()
        
        # Test end-to-end workflows
        await self.test_end_to_end_research_workflow()
        
        # Generate summary
        return self.generate_summary()

async def main():
    tester = AgentExecutionTester()
    summary = await tester.run_all_tests()
    
    # Save results
    import json
    with open("agent_execution_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: agent_execution_results.json")
    print("\n🎉 Agent execution testing completed!")

if __name__ == "__main__":
    asyncio.run(main())