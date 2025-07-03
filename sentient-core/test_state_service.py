"""Comprehensive tests for the State Management Service

This test suite validates all functionality of the StateService including:
- Agent state management
- Workflow state tracking
- Conversation state handling
- Checkpoint creation and recovery
- Event listeners
- Performance tracking
- Database persistence
"""

import asyncio
import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

# Import the state service components
from core.services.state_service import (
    StateService,
    AgentState,
    WorkflowState,
    ConversationState,
    StateCheckpoint,
    AgentStatus,
    WorkflowStatus,
    state_service_context
)

class TestStateService:
    """Test suite for StateService functionality"""
    
    @pytest.fixture
    async def temp_state_service(self):
        """Create a temporary state service for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            async with state_service_context(db_path, checkpoint_interval=0) as service:
                yield service
        finally:
            # Cleanup - handle Windows file locking issues
            try:
                if os.path.exists(db_path):
                    os.unlink(db_path)
            except PermissionError:
                # On Windows, the file might still be locked
                import time
                time.sleep(0.1)
                try:
                    os.unlink(db_path)
                except PermissionError:
                    pass  # Ignore cleanup errors in tests
    
    async def test_agent_state_management(self, temp_state_service):
        """Test agent state creation, updates, and retrieval"""
        service = temp_state_service
        
        # Test creating new agent state
        agent_state = await service.update_agent_state(
            "test_agent_1",
            status="active",
            current_task="test_task",
            metadata={"test_key": "test_value"}
        )
        
        assert agent_state.agent_id == "test_agent_1"
        assert agent_state.status == AgentStatus.ACTIVE
        assert agent_state.current_task == "test_task"
        assert agent_state.metadata["test_key"] == "test_value"
        assert agent_state.last_activity is not None
        
        # Test retrieving agent state
        retrieved_state = await service.get_agent_state("test_agent_1")
        assert retrieved_state is not None
        assert retrieved_state.agent_id == "test_agent_1"
        assert retrieved_state.status == AgentStatus.ACTIVE
        
        # Test updating existing agent state
        updated_state = await service.update_agent_state(
            "test_agent_1",
            status="busy",
            current_task="new_task"
        )
        
        assert updated_state.status == AgentStatus.BUSY
        assert updated_state.current_task == "new_task"
        assert updated_state.metadata["test_key"] == "test_value"  # Should persist
        
        # Test getting all agent states
        all_states = await service.get_all_agent_states()
        assert len(all_states) == 1
        assert "test_agent_1" in all_states
        
        print("✓ Agent state management tests passed")
    
    async def test_workflow_state_management(self, temp_state_service):
        """Test workflow state creation, updates, and progress tracking"""
        service = temp_state_service
        
        # Test creating new workflow state
        workflow_state = await service.update_workflow_state(
            "test_workflow_1",
            status="running",
            current_step="step_1",
            progress=25.0,
            steps_completed=["init"],
            steps_remaining=["step_1", "step_2", "finish"]
        )
        
        assert workflow_state.workflow_id == "test_workflow_1"
        assert workflow_state.status == WorkflowStatus.RUNNING
        assert workflow_state.current_step == "step_1"
        assert workflow_state.progress == 25.0
        assert "init" in workflow_state.steps_completed
        
        # Test workflow progress update
        updated_workflow = await service.update_workflow_state(
            "test_workflow_1",
            current_step="step_2",
            progress=75.0,
            steps_completed=["init", "step_1"],
            steps_remaining=["step_2", "finish"]
        )
        
        assert updated_workflow.current_step == "step_2"
        assert updated_workflow.progress == 75.0
        assert len(updated_workflow.steps_completed) == 2
        
        # Test auto-completion when progress reaches 100%
        completed_workflow = await service.update_workflow_state(
            "test_workflow_1",
            progress=100.0,
            steps_completed=["init", "step_1", "step_2", "finish"],
            steps_remaining=[]
        )
        
        assert completed_workflow.status == WorkflowStatus.COMPLETED
        assert completed_workflow.completed_at is not None
        
        # Test retrieving workflow status
        retrieved_workflow = await service.get_workflow_status("test_workflow_1")
        assert retrieved_workflow is not None
        assert retrieved_workflow.status == WorkflowStatus.COMPLETED
        
        # Test getting all workflow states
        all_workflows = await service.get_all_workflow_states()
        assert len(all_workflows) == 1
        assert "test_workflow_1" in all_workflows
        
        print("✓ Workflow state management tests passed")
    
    async def test_conversation_state_management(self, temp_state_service):
        """Test conversation state creation and updates"""
        service = temp_state_service
        
        # Test creating conversation state
        conv_state = await service.update_conversation_state(
            "test_conversation_1",
            user_id="user_123",
            context={"topic": "AI development", "session_id": "sess_456"},
            active_agents=["agent_1", "agent_2"]
        )
        
        assert conv_state.conversation_id == "test_conversation_1"
        assert conv_state.user_id == "user_123"
        assert conv_state.context["topic"] == "AI development"
        assert "agent_1" in conv_state.active_agents
        assert conv_state.last_interaction is not None
        
        # Test updating conversation state
        updated_conv = await service.update_conversation_state(
            "test_conversation_1",
            active_agents=["agent_1", "agent_2", "agent_3"],
            context={"topic": "AI development", "session_id": "sess_456", "step": "research"}
        )
        
        assert len(updated_conv.active_agents) == 3
        assert updated_conv.context["step"] == "research"
        
        # Test retrieving conversation state
        retrieved_conv = await service.get_conversation_state()
        assert retrieved_conv is not None
        assert retrieved_conv.conversation_id == "test_conversation_1"
        
        print("✓ Conversation state management tests passed")
    
    async def test_checkpoint_and_recovery(self, temp_state_service):
        """Test checkpoint creation and state recovery"""
        service = temp_state_service
        
        # Create some initial state
        await service.update_agent_state("agent_1", status="active", current_task="task_1")
        await service.update_agent_state("agent_2", status="busy", current_task="task_2")
        await service.update_workflow_state("workflow_1", status="running", progress=50.0)
        await service.update_conversation_state("conv_1", user_id="user_1")
        
        # Create checkpoint
        checkpoint = await service.checkpoint_state(
            "test_checkpoint_1",
            metadata={"test": "checkpoint", "version": "1.0"}
        )
        
        assert checkpoint.checkpoint_id == "test_checkpoint_1"
        assert len(checkpoint.agent_states) == 2
        assert len(checkpoint.workflow_states) == 1
        assert checkpoint.metadata["test"] == "checkpoint"
        
        # Modify state after checkpoint
        await service.update_agent_state("agent_1", status="idle")
        await service.update_workflow_state("workflow_1", progress=100.0)
        
        # Verify state has changed
        agent_state = await service.get_agent_state("agent_1")
        assert agent_state.status == AgentStatus.IDLE
        
        workflow_state = await service.get_workflow_status("workflow_1")
        assert workflow_state.status == WorkflowStatus.COMPLETED
        
        # Restore from checkpoint
        restore_success = await service.restore_from_checkpoint("test_checkpoint_1")
        assert restore_success is True
        
        # Verify state has been restored
        restored_agent = await service.get_agent_state("agent_1")
        assert restored_agent.status == AgentStatus.ACTIVE
        
        restored_workflow = await service.get_workflow_status("workflow_1")
        assert restored_workflow.status == WorkflowStatus.RUNNING
        assert restored_workflow.progress == 50.0
        
        # Test restoring from non-existent checkpoint
        restore_fail = await service.restore_from_checkpoint("non_existent")
        assert restore_fail is False
        
        print("✓ Checkpoint and recovery tests passed")
    
    async def test_event_listeners(self, temp_state_service):
        """Test event listener functionality"""
        service = temp_state_service
        
        # Track events
        events_received = []
        
        def agent_listener(agent_id, state):
            events_received.append(("agent", agent_id, state.status.value))
        
        def workflow_listener(workflow_id, state):
            events_received.append(("workflow", workflow_id, state.status.value))
        
        async def conversation_listener(conv_id, state):
            events_received.append(("conversation", conv_id, len(state.active_agents)))
        
        # Add listeners
        service.add_state_listener("agent_update", agent_listener)
        service.add_state_listener("workflow_update", workflow_listener)
        service.add_state_listener("conversation_update", conversation_listener)
        
        # Trigger events
        await service.update_agent_state("agent_1", status="active")
        await service.update_workflow_state("workflow_1", status="running")
        await service.update_conversation_state("conv_1", active_agents=["agent_1"])
        
        # Allow time for async listeners
        await asyncio.sleep(0.1)
        
        # Verify events were received
        assert len(events_received) == 3
        assert ("agent", "agent_1", "active") in events_received
        assert ("workflow", "workflow_1", "running") in events_received
        assert ("conversation", "conv_1", 1) in events_received
        
        # Test removing listeners
        service.remove_state_listener("agent_update", agent_listener)
        
        # Trigger another event
        await service.update_agent_state("agent_2", status="busy")
        await asyncio.sleep(0.1)
        
        # Should still be 3 events (agent listener was removed)
        assert len(events_received) == 3
        
        print("✓ Event listener tests passed")
    
    async def test_performance_tracking(self, temp_state_service):
        """Test performance statistics tracking"""
        service = temp_state_service
        
        # Perform some operations
        await service.update_agent_state("agent_1", status="active")
        await service.update_agent_state("agent_2", status="busy")
        await service.update_workflow_state("workflow_1", status="running")
        await service.checkpoint_state("perf_checkpoint")
        
        # Get performance stats
        stats = await service.get_performance_stats()
        
        assert stats["state_updates"] >= 3  # At least 3 state updates
        assert stats["checkpoints_created"] >= 1  # At least 1 checkpoint
        assert stats["active_agents"] == 2  # 2 active agents
        assert stats["active_workflows"] == 1  # 1 active workflow
        assert "avg_update_time" in stats
        assert stats["avg_update_time"] >= 0
        
        print("✓ Performance tracking tests passed")
    
    async def test_state_cleanup(self, temp_state_service):
        """Test cleanup of old states"""
        service = temp_state_service
        
        # Create some states
        await service.update_agent_state("agent_1", status="idle")
        await service.update_agent_state("agent_2", status="active")
        await service.update_workflow_state("workflow_1", status="completed")
        await service.update_workflow_state("workflow_2", status="running")
        
        # Manually set old timestamps for testing
        old_time = datetime.utcnow() - timedelta(hours=25)
        service.agent_states["agent_1"].last_activity = old_time
        service.workflow_states["workflow_1"].completed_at = old_time
        
        # Verify initial state count
        assert len(service.agent_states) == 2
        assert len(service.workflow_states) == 2
        
        # Run cleanup
        await service.cleanup_old_states(max_age_hours=24)
        
        # Verify cleanup results
        assert len(service.agent_states) == 1  # Only active agent should remain
        assert len(service.workflow_states) == 1  # Only running workflow should remain
        assert "agent_2" in service.agent_states
        assert "workflow_2" in service.workflow_states
        
        print("✓ State cleanup tests passed")
    
    async def test_database_persistence(self, temp_state_service):
        """Test database persistence and loading"""
        service = temp_state_service
        
        # Create some state
        await service.update_agent_state(
            "persistent_agent",
            status="active",
            current_task="persistent_task",
            metadata={"persistent": True}
        )
        
        await service.update_workflow_state(
            "persistent_workflow",
            status="running",
            progress=75.0,
            metadata={"persistent": True}
        )
        
        # Stop and restart service to test persistence
        db_path = service.db_path
        await service.stop()
        
        # Create new service with same database
        new_service = StateService(str(db_path), checkpoint_interval=0)
        await new_service.start()
        
        try:
            # Verify state was loaded from database
            loaded_agent = await new_service.get_agent_state("persistent_agent")
            assert loaded_agent is not None
            assert loaded_agent.status == AgentStatus.ACTIVE
            assert loaded_agent.current_task == "persistent_task"
            assert loaded_agent.metadata["persistent"] is True
            
            loaded_workflow = await new_service.get_workflow_status("persistent_workflow")
            assert loaded_workflow is not None
            assert loaded_workflow.status == WorkflowStatus.RUNNING
            assert loaded_workflow.progress == 75.0
            assert loaded_workflow.metadata["persistent"] is True
            
            print("✓ Database persistence tests passed")
            
        finally:
            await new_service.stop()
    
    async def test_error_handling(self, temp_state_service):
        """Test error handling and edge cases"""
        service = temp_state_service
        
        # Test getting non-existent states
        non_existent_agent = await service.get_agent_state("non_existent")
        assert non_existent_agent is None
        
        non_existent_workflow = await service.get_workflow_status("non_existent")
        assert non_existent_workflow is None
        
        # Test invalid status values (should be handled gracefully)
        try:
            await service.update_agent_state("test_agent", status="invalid_status")
            assert False, "Should have raised an exception"
        except ValueError:
            pass  # Expected
        
        # Test updating with invalid field names (should be ignored)
        state = await service.update_agent_state(
            "test_agent",
            status="active",
            invalid_field="should_be_ignored"
        )
        assert not hasattr(state, "invalid_field")
        
        print("✓ Error handling tests passed")

# Mock test runner for standalone execution
async def run_tests():
    """Run all tests manually with fresh service for each test"""
    print("Running State Service Tests...\n")
    
    test_instance = TestStateService()
    test_methods = [
        ("test_agent_state_management", test_instance.test_agent_state_management),
        ("test_workflow_state_management", test_instance.test_workflow_state_management),
        ("test_conversation_state_management", test_instance.test_conversation_state_management),
        ("test_checkpoint_and_recovery", test_instance.test_checkpoint_and_recovery),
        ("test_event_listeners", test_instance.test_event_listeners),
        ("test_performance_tracking", test_instance.test_performance_tracking),
        ("test_state_cleanup", test_instance.test_state_cleanup),
        ("test_database_persistence", test_instance.test_database_persistence),
        ("test_error_handling", test_instance.test_error_handling),
    ]
    
    for test_name, test_method in test_methods:
        print(f"Running {test_name}...")
        
        # Create fresh temporary database for each test
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            async with state_service_context(db_path, checkpoint_interval=0) as service:
                await test_method(service)
                
        finally:
            # Cleanup - handle Windows file locking issues
            try:
                if os.path.exists(db_path):
                    os.unlink(db_path)
            except PermissionError:
                # On Windows, the file might still be locked
                import time
                time.sleep(0.1)
                try:
                    os.unlink(db_path)
                except PermissionError:
                    print(f"Warning: Could not delete temporary database file: {db_path}")
    
    print("\n🎉 All State Service tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(run_tests())