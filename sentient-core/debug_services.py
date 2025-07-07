#!/usr/bin/env python3
"""
Debug script to test individual service initialization
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_memory_service():
    """Test memory service initialization"""
    try:
        logger.info("Testing Memory Service...")
        from app.services.memory_service import MemoryService
        memory_service = MemoryService()
        logger.info("✓ Memory Service: OK")
        return True
    except Exception as e:
        logger.error(f"✗ Memory Service: FAILED - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_state_manager():
    """Test state manager initialization"""
    try:
        logger.info("Testing State Manager...")
        from app.services.state_manager import EnhancedStateManager
        state_manager = EnhancedStateManager()
        logger.info("✓ State Manager: OK")
        return True
    except Exception as e:
        logger.error(f"✗ State Manager: FAILED - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_sse_manager():
    """Test SSE manager initialization"""
    try:
        logger.info("Testing SSE Manager...")
        from app.services.sse_manager import SSEConnectionManager
        sse_manager = SSEConnectionManager()
        logger.info("✓ SSE Manager: OK")
        return True
    except Exception as e:
        logger.error(f"✗ SSE Manager: FAILED - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_llm_service():
    """Test LLM service initialization"""
    try:
        logger.info("Testing LLM Service...")
        from app.services.llm_service import EnhancedLLMService
        llm_service = EnhancedLLMService()
        logger.info("✓ LLM Service: OK")
        return True
    except Exception as e:
        logger.error(f"✗ LLM Service: FAILED - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_workflow_service():
    """Test Workflow Service initialization"""
    try:
        logger.info("Testing Workflow Service...")
        from app.services.memory_service import MemoryService
        from app.services.state_manager import EnhancedStateManager
        from app.services.llm_service import EnhancedLLMService
        from app.services.sse_manager import SSEConnectionManager
        from app.services.workflow_service import WorkflowOrchestrator
        
        memory_service = MemoryService()
        state_manager = EnhancedStateManager()
        llm_service = EnhancedLLMService()
        sse_manager = SSEConnectionManager()
        
        workflow_service = WorkflowOrchestrator(
            memory_service=memory_service,
            state_manager=state_manager,
            llm_service=llm_service,
            sse_manager=sse_manager
        )
        logger.info("✓ Workflow Service: OK")
        return True
    except Exception as e:
        logger.error(f"✗ Workflow Service: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_research_service():
    """Test research service initialization"""
    try:
        logger.info("Testing Research Service...")
        from app.services.memory_service import MemoryService
        from app.services.llm_service import EnhancedLLMService
        from app.services.sse_manager import SSEConnectionManager
        from app.services.research_service import EnhancedResearchService, ResearchConfig
        
        memory_service = MemoryService()
        llm_service = EnhancedLLMService()
        sse_manager = SSEConnectionManager()
        config = ResearchConfig()
        
        research_service = EnhancedResearchService(
            config=config,
            memory_service=memory_service,
            llm_service=llm_service,
            sse_manager=sse_manager
        )
        logger.info("✓ Research Service: OK")
        return True
    except Exception as e:
        logger.error(f"✗ Research Service: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_service():
    """Test agent service initialization"""
    try:
        logger.info("Testing Agent Service...")
        from app.services.memory_service import MemoryService
        from app.services.state_manager import EnhancedStateManager
        from app.services.llm_service import EnhancedLLMService
        from app.services.sse_manager import SSEConnectionManager
        from app.services.workflow_service import WorkflowOrchestrator
        from app.services.research_service import EnhancedResearchService, ResearchConfig
        from app.services.agent_service import AgentService
        
        memory_service = MemoryService()
        state_manager = EnhancedStateManager()
        llm_service = EnhancedLLMService()
        sse_manager = SSEConnectionManager()
        
        workflow_service = WorkflowOrchestrator(
            memory_service=memory_service,
            state_manager=state_manager,
            llm_service=llm_service,
            sse_manager=sse_manager
        )
        
        research_config = ResearchConfig()
        research_service = EnhancedResearchService(
            config=research_config,
            memory_service=memory_service,
            llm_service=llm_service,
            sse_manager=sse_manager
        )
        
        agent_service = AgentService(
        memory_service=memory_service,
        llm_service=llm_service,
        sse_manager=sse_manager,
        workflow_service=workflow_service,
        research_service=research_service
    )
        logger.info("✓ Agent Service: OK")
        return True
    except Exception as e:
        logger.error(f"✗ Agent Service: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all service tests"""
    logger.info("🔍 Starting service initialization tests...")
    
    tests = [
        ("Memory Service", test_memory_service),
        ("State Manager", test_state_manager),
        ("SSE Manager", test_sse_manager),
        ("LLM Service", test_llm_service),
        ("Workflow Service", test_workflow_service),
        ("Research Service", test_research_service),
        ("Agent Service", test_agent_service),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All services can be initialized successfully!")
    else:
        logger.error("💥 Some services failed to initialize")

if __name__ == "__main__":
    asyncio.run(main())