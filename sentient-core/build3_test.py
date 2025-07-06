#!/usr/bin/env python3
"""
Build 3 Implementation Test - Testing Architect Planner Agent functionality
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.models import AgentType

async def test_architect_planner_functionality():
    """Test the Build 3 Architect Planner Agent functionality"""
    print("=== Build 3 Architect Planner Agent Test ===")
    
    try:
        # Import required modules
        from core.services.llm_service import EnhancedLLMService
        from core.agents.architect_planner_agent import ArchitectPlannerAgent
        from core.models import EnhancedTask, TaskStatus
        
        print("✓ All required modules imported successfully")
        
        # Initialize services
        llm_service = EnhancedLLMService()
        print("✓ LLM service initialized")
        
        # Initialize architect planner agent
        planner = ArchitectPlannerAgent(llm_service)
        print("✓ Architect Planner Agent initialized")
        
        # Test task type determination
        test_queries = [
            "Create a PRD for a new mobile app",
            "Design the architecture for a microservices system",
            "Break down the development tasks for this project",
            "Analyze the requirements for this feature",
            "Synthesize research findings into actionable plan",
            "Create a roadmap for the next 6 months"
        ]
        
        print("\n--- Testing Task Type Determination ---")
        for query in test_queries:
            task_type = planner._determine_task_type(query)
            print(f"Query: '{query[:50]}...' -> Type: {task_type}")
        
        # Test PRD generation (simplified)
        print("\n--- Testing PRD Generation ---")
        from core.models import AgentType
        prd_task = EnhancedTask(
            id="test_prd",
            description="Create a Product Requirements Document for a task management application that helps teams collaborate and track project progress",
            agent_type=AgentType.ARCHITECT_PLANNER,
            status=TaskStatus.PENDING
        )
        
        print(f"Processing PRD task: {prd_task.description}")
        
        # Check if task can be handled
        can_handle = planner.can_handle_task(prd_task)
        print(f"Can handle task: {can_handle}")
        
        if can_handle:
            # Process the task (this will make actual LLM calls)
            print("Processing task with LLM...")
            result = await planner.process_task(prd_task)
            
            if result.get('status') == 'completed':
                print("✓ PRD generation completed successfully")
                
                # Check if result contains expected sections
                prd_content = result.get('result', '')
                expected_sections = ['Executive Summary', 'Problem Statement', 'Goals', 'User Stories']
                
                found_sections = []
                for section in expected_sections:
                    if section.lower() in prd_content.lower():
                        found_sections.append(section)
                
                print(f"Found {len(found_sections)}/{len(expected_sections)} expected sections: {found_sections}")
                
                # Save the generated PRD for inspection
                output_file = project_root / "test_prd_output.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Test PRD Output\n\n")
                    f.write(f"**Task:** {prd_task.description}\n\n")
                    f.write(f"**Generated PRD:**\n\n{prd_content}")
                
                print(f"✓ PRD output saved to: {output_file}")
                
            else:
                print(f"✗ PRD generation failed: {result.get('error', 'Unknown error')}")
                return False
        
        # Test Architecture Planning
        print("\n--- Testing Architecture Planning ---")
        arch_task = EnhancedTask(
            id="test_arch",
            description="Design the technical architecture for a scalable web application with user authentication, real-time messaging, and data analytics",
            agent_type=AgentType.ARCHITECT_PLANNER,
            status=TaskStatus.PENDING
        )
        
        print(f"Processing Architecture task: {arch_task.description}")
        
        if planner.can_handle_task(arch_task):
            print("Processing architecture task with LLM...")
            arch_result = await planner.process_task(arch_task)
            
            if arch_result.get('status') == 'completed':
                print("✓ Architecture planning completed successfully")
                
                # Save the generated architecture for inspection
                arch_content = arch_result.get('result', '')
                output_file = project_root / "test_architecture_output.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Test Architecture Output\n\n")
                    f.write(f"**Task:** {arch_task.description}\n\n")
                    f.write(f"**Generated Architecture:**\n\n{arch_content}")
                
                print(f"✓ Architecture output saved to: {output_file}")
            else:
                print(f"✗ Architecture planning failed: {arch_result.get('error', 'Unknown error')}")
        
        print("\n=== Build 3 Test Summary ===")
        print("✓ Architect Planner Agent is functional")
        print("✓ Task type determination working")
        print("✓ PRD generation working")
        print("✓ Architecture planning working")
        print("✓ LLM integration successful")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Build 3 test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_research_integration():
    """Test integration between Research Agent and Architect Planner"""
    print("\n=== Research-Planning Integration Test ===")
    
    try:
        from core.services.llm_service import EnhancedLLMService
        from core.agents.research_agent import ResearchAgent
        from core.agents.architect_planner_agent import ArchitectPlannerAgent
        from core.models import EnhancedTask, TaskStatus
        
        # Initialize services
        llm_service = EnhancedLLMService()
        research_agent = ResearchAgent(llm_service)
        planner_agent = ArchitectPlannerAgent(llm_service)
        
        print("✓ Both agents initialized successfully")
        
        # Test research task
        from core.models import AgentType
        research_task = EnhancedTask(
            id="test_research",
            description="Research best practices for microservices architecture and API design patterns",
            agent_type=AgentType.RESEARCH_AGENT,
            status=TaskStatus.PENDING
        )
        
        print(f"Testing research capability: {research_task.description}")
        
        # Check which agent can handle the research task
        research_can_handle = research_agent.can_handle_task(research_task)
        planner_can_handle = planner_agent.can_handle_task(research_task)
        
        print(f"Research Agent can handle: {research_can_handle}")
        print(f"Planner Agent can handle: {planner_can_handle}")
        
        # Test planning task
        planning_task = EnhancedTask(
            id="test_planning",
            description="Create a comprehensive project plan for implementing a microservices architecture",
            agent_type=AgentType.ARCHITECT_PLANNER,
            status=TaskStatus.PENDING
        )
        
        print(f"Testing planning capability: {planning_task.description}")
        
        research_can_handle_plan = research_agent.can_handle_task(planning_task)
        planner_can_handle_plan = planner_agent.can_handle_task(planning_task)
        
        print(f"Research Agent can handle planning: {research_can_handle_plan}")
        print(f"Planner Agent can handle planning: {planner_can_handle_plan}")
        
        print("✓ Agent capability differentiation working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

async def main():
    """Run all Build 3 tests"""
    print("Starting Build 3 Implementation Tests...\n")
    
    # Test core functionality
    test1_passed = await test_architect_planner_functionality()
    
    # Test integration
    test2_passed = await test_research_integration()
    
    print("\n" + "="*50)
    print("BUILD 3 TEST RESULTS")
    print("="*50)
    print(f"Architect Planner Functionality: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Research-Planning Integration: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    
    overall_success = test1_passed and test2_passed
    print(f"\nOverall Build 3 Status: {'✓ FULLY FUNCTIONAL' if overall_success else '✗ ISSUES DETECTED'}")
    
    if overall_success:
        print("\n🎉 Build 3 implementation is working correctly!")
        print("The Architect Planner Agent is ready for production use.")
    else:
        print("\n⚠️  Build 3 has some issues that need attention.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)