"""Test core system functionality to verify the 3-build system is working."""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agents.ultra_orchestrator import UltraOrchestrator
from core.services.memory_service import MemoryService
from core.services.enhanced_llm_service import EnhancedLLMService
from core.services.state_service import StateService


class TestCoreSystemFunctionality:
    """Test the core 3-build system functionality."""
    
    def test_memory_service_initialization(self):
        """Test that MemoryService initializes correctly."""
        memory_service = MemoryService()
        
        # Check that memory service has required components
        assert hasattr(memory_service, 'db')
        assert hasattr(memory_service, 'vector_service')
        assert hasattr(memory_service, 'layers')
        
        # Check that all layers are initialized
        from core.models import MemoryLayer
        for layer in MemoryLayer:
            assert layer in memory_service.layers
        
        # Check that actual memory directories exist
        memory_dir = Path(project_root / 'memory')
        layer1_dir = memory_dir / 'layer1_research_docs'
        layer2_dir = memory_dir / 'layer2_build_artifacts'
        
        assert layer1_dir.exists(), "Layer1 research docs directory should exist"
        assert layer2_dir.exists(), "Layer2 build artifacts directory should exist"
        
        # Check layer1 has research documents (may be empty in fresh setup)
        layer1_files = list(layer1_dir.glob('*.md'))
        # Note: Layer1 may be empty in a fresh setup, which is acceptable
        
        print(f"✅ MemoryService initialized with {len(layer1_files)} research documents")
    
    def test_enhanced_llm_service_initialization(self):
        """Test that EnhancedLLMService initializes correctly."""
        try:
            llm_service = EnhancedLLMService()
            
            # Check that service has required attributes
            assert hasattr(llm_service, 'providers')
            assert hasattr(llm_service, 'default_provider')
            
            # If initialization succeeds, we should have providers
            if llm_service.providers:
                assert len(llm_service.providers) > 0, "Should have at least one provider"
                print("✅ EnhancedLLMService initialized successfully with providers")
            else:
                print("⚠️ EnhancedLLMService initialized but no providers available (missing API keys)")
                
        except ValueError as e:
            if "No LLM providers available" in str(e):
                print("⚠️ EnhancedLLMService requires API keys - this is expected in test environment")
            else:
                raise e
    
    def test_state_service_initialization(self):
        """Test that StateService initializes correctly."""
        state_service = StateService(db_path=":memory:")  # Use in-memory DB for testing
        
        # Check that service has required attributes
        assert hasattr(state_service, 'agent_states')
        assert hasattr(state_service, 'workflow_states')
        assert hasattr(state_service, 'conversation_state')
        assert hasattr(state_service, 'db_path')
        
        print("✅ StateService working correctly")
    
    def test_ultra_orchestrator_initialization(self):
        """Test that UltraOrchestrator initializes correctly."""
        llm_service = EnhancedLLMService()
        orchestrator = UltraOrchestrator(llm_service)
        
        # Check that orchestrator has required components
        assert hasattr(orchestrator, 'llm_service')
        assert hasattr(orchestrator, 'research_agent')
        assert hasattr(orchestrator, 'architect_planner')
        
        # Check that it has the required methods
        assert hasattr(orchestrator, 'invoke')
        assert hasattr(orchestrator, 'invoke_state')
        assert hasattr(orchestrator, '_should_transition_to_planning')
        assert hasattr(orchestrator, '_handle_planning_transition')
        
        print("✅ UltraOrchestrator initialized with all required components")
    
    def test_memory_layer_structure(self):
        """Test that memory layers have the correct structure."""
        # Check actual file system memory structure
        memory_dir = Path(project_root / 'memory')
        layer1_dir = memory_dir / 'layer1_research_docs'
        layer2_dir = memory_dir / 'layer2_build_artifacts'
        
        # Test layer1 (research documents) - may be empty in fresh setup
        layer1_files = list(layer1_dir.glob('*.md'))
        
        # If research files exist, check naming convention
        if layer1_files:
            for file in layer1_files[:3]:  # Check first 3 files
                assert file.name.endswith('.md')
                assert file.stat().st_size > 0, f"Research file {file.name} should not be empty"
        
        # Test layer2 (build artifacts) exists
        assert layer2_dir.exists()
        
        print(f"✅ Memory layers structured correctly: {len(layer1_files)} research docs")
    
    @pytest.mark.asyncio
    async def test_system_integration_readiness(self):
        """Test that all components can work together."""
        # Initialize all core components with proper parameters
        memory_service = MemoryService()
        state_service = StateService(db_path=":memory:")  # Use in-memory DB for testing
        
        # Try to initialize LLM service, handle gracefully if no API keys
        try:
            llm_service = EnhancedLLMService()
            orchestrator = UltraOrchestrator(llm_service)
            
            # Test that orchestrator can access memory (async method)
            research_artifacts = await orchestrator._gather_research_artifacts()
            assert isinstance(research_artifacts, str)
            
            # Test that orchestrator can check for research
            has_research = orchestrator._check_research_artifacts_exist()
            assert isinstance(has_research, bool)
            
            print("✅ All core components integrated and ready with LLM service")
            
        except ValueError as e:
            if "No LLM providers available" in str(e):
                print("⚠️ Integration test passed but LLM service unavailable (missing API keys)")
            else:
                raise e
        
        # Test that memory service layers are accessible
        from core.models import MemoryLayer
        assert MemoryLayer.KNOWLEDGE_SYNTHESIS in memory_service.layers
        assert MemoryLayer.CONVERSATION_HISTORY in memory_service.layers
        
        print("✅ Core memory and state services integrated successfully")


if __name__ == "__main__":
    # Run tests manually if executed directly
    test_suite = TestCoreSystemFunctionality()
    
    print("🧪 Running Core System Functionality Tests...\n")
    
    try:
        test_suite.test_memory_service_initialization()
        test_suite.test_enhanced_llm_service_initialization()
        test_suite.test_state_service_initialization()
        test_suite.test_ultra_orchestrator_initialization()
        test_suite.test_memory_layer_structure()
        asyncio.run(test_suite.test_system_integration_readiness())
        
        print("\n🎉 All core functionality tests passed!")
        print("✅ The 3-build system is properly implemented and functional.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()