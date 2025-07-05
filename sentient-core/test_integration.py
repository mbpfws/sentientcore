#!/usr/bin/env python3
"""
Integration Test Script

This script tests the core system integration to verify that all components
can be initialized and work together properly.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_core_services():
    """Test core services initialization."""
    print("\n🔧 Testing Core Services Integration...")
    
    try:
        # Test StateService
        from core.services.state_service import StateService
        state_service = StateService()
        await state_service.start()  # Use start() method
        print("✅ StateService initialized successfully")
        
        # Test EnhancedLLMService
        from core.services.llm_service import EnhancedLLMService
        llm_service = EnhancedLLMService()  # Initializes in constructor
        print("✅ EnhancedLLMService initialized successfully")
        
        # Test WorkflowOrchestrator
        from core.orchestration import initialize_workflow_orchestrator
        orchestrator = await initialize_workflow_orchestrator(state_service, llm_service)
        print("✅ WorkflowOrchestrator initialized successfully")
        
        return True, state_service, llm_service, orchestrator
        
    except Exception as e:
        print(f"❌ Core services initialization failed: {e}")
        logger.exception("Core services initialization error")
        return False, None, None, None

async def test_agent_system(state_service, llm_service):
    """Test agent system integration."""
    print("\n🤖 Testing Agent System Integration...")
    
    try:
        from core.agents.integration import AgentSystemIntegration
        
        agent_system = AgentSystemIntegration()
        success = await agent_system.initialize(state_service, llm_service)
        
        if success:
            print("✅ Agent system initialized successfully")
            print(f"   - Agents created: {list(agent_system.agents.keys())}")
            return True, agent_system
        else:
            print("❌ Agent system initialization failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Agent system initialization failed: {e}")
        logger.exception("Agent system initialization error")
        return False, None

async def test_graph_integration(state_service, llm_service):
    """Test graph integration manager."""
    print("\n📊 Testing Graph Integration...")
    
    try:
        from core.graphs.graph_integration_manager import GraphIntegrationManager
        
        # Create a mock memory service for testing
        class MockMemoryService:
            def store_conversation_memory(self, **kwargs):
                pass
        
        memory_service = MockMemoryService()
        
        graph_manager = GraphIntegrationManager(
            llm_service=llm_service,
            memory_service=memory_service,
            state_service=state_service
        )
        
        # Test session creation
        session = graph_manager.create_session("test_session")
        print("✅ Graph integration manager initialized successfully")
        print(f"   - Available graphs: {list(graph_manager.graphs.keys())}")
        print(f"   - Test session created: {session.session_id}")
        
        return True, graph_manager
        
    except Exception as e:
        print(f"❌ Graph integration initialization failed: {e}")
        logger.exception("Graph integration initialization error")
        return False, None

async def test_workflow_execution(orchestrator):
    """Test basic workflow execution."""
    print("\n🔄 Testing Workflow Execution...")
    
    try:
        from core.models import AppState
        
        # Create test app state
        app_state = AppState()
        
        # Test workflow execution with a simple request
        result = await orchestrator.execute_workflow(
            user_input="Hello, test the system",
            app_state=app_state,
            workflow_mode="intelligent",
            research_mode="knowledge"
        )
        
        print("✅ Workflow execution completed successfully")
        print(f"   - Result keys: {list(result.keys())}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        logger.exception("Workflow execution error")
        return False, None

async def test_api_endpoints():
    """Test API endpoints availability."""
    print("\n🌐 Testing API Endpoints...")
    
    try:
        # Test that FastAPI app can be imported
        from app.api.app import app
        
        print("✅ FastAPI app can be imported successfully")
        print(f"   - App title: {getattr(app, 'title', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        logger.exception("API endpoints test error")
        return False

async def main():
    """Main test function."""
    print("🚀 Starting SentientCore Integration Test")
    print("=" * 50)
    
    # Test core services
    services_ok, state_service, llm_service, orchestrator = await test_core_services()
    if not services_ok:
        print("\n❌ Core services test failed. Stopping tests.")
        return False
    
    # Test agent system
    agent_ok, agent_system = await test_agent_system(state_service, llm_service)
    
    # Test graph integration
    graph_ok, graph_manager = await test_graph_integration(state_service, llm_service)
    
    # Test workflow execution
    workflow_ok, workflow_result = await test_workflow_execution(orchestrator)
    
    # Test API endpoints
    api_ok = await test_api_endpoints()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Integration Test Summary:")
    print(f"   Core Services: {'✅ PASS' if services_ok else '❌ FAIL'}")
    print(f"   Agent System: {'✅ PASS' if agent_ok else '❌ FAIL'}")
    print(f"   Graph Integration: {'✅ PASS' if graph_ok else '❌ FAIL'}")
    print(f"   Workflow Execution: {'✅ PASS' if workflow_ok else '❌ FAIL'}")
    print(f"   API Endpoints: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    overall_success = all([services_ok, agent_ok, graph_ok, workflow_ok, api_ok])
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED! System is ready for end-to-end testing.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    return overall_success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.exception("Unexpected test error")
        sys.exit(1)