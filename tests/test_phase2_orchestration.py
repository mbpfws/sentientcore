"""
Test suite for the Phase 2 implementation of the Sentient Agentic Framework.
Validates the core decision-making loop between the UltraOrchestrator and MonitoringAgent.
"""

import pytest
import os
from dotenv import load_dotenv

# The path resolution for imports is handled by tests/conftest.py
load_dotenv()

from core.models import AppState, Message, AgentType
from graphs.sentient_workflow_graph import app as sentient_workflow_app

def run_workflow_with_prompt(prompt: str) -> AppState:
    """Helper function to run the workflow with a single user prompt."""
    initial_state = AppState(
        messages=[Message(sender="user", content=prompt)]
    )
    final_state_dict = sentient_workflow_app.invoke(initial_state)
    return AppState(**final_state_dict)

def test_vague_request_clarification():
    """
    Tests if the UltraOrchestrator correctly identifies a vague request
    and asks for clarification.
    """
    prompt = "I want to build an app."
    final_state = run_workflow_with_prompt(prompt)

    # 1. Check the orchestrator's decision
    assert final_state.next_action == "request_clarification", \
        "Orchestrator should have decided to request clarification."

    # 2. Check the assistant's response
    assistant_response = final_state.messages[-1].content
    assert "purpose" in assistant_response.lower() or "what kind" in assistant_response.lower(), \
        "Assistant should have asked a clarifying question about the app's purpose."
        
    # 3. Check the monitor's log
    monitor_log = next((log for log in reversed(final_state.logs) if log.source == "MonitoringAgent"), None)
    assert monitor_log is not None, "MonitoringAgent should have logged an action."
    assert "request_clarification" in monitor_log.message, \
        "Monitor should have logged the 'request_clarification' decision."

def test_off_topic_redirect():
    """
    Tests if the UltraOrchestrator correctly identifies an off-topic request
    and redirects the conversation.
    """
    prompt = "What is the capital of France?"
    final_state = run_workflow_with_prompt(prompt)

    # 1. Check the orchestrator's decision
    assert final_state.next_action == "redirect_off_topic", \
        "Orchestrator should have decided to redirect the conversation."

    # 2. Check the assistant's response
    assistant_response = final_state.messages[-1].content
    assert "software" in assistant_response.lower() or "development" in assistant_response.lower(), \
        "Assistant should have redirected the conversation back to development."
        
    # 3. Check the monitor's log
    monitor_log = next((log for log in reversed(final_state.logs) if log.source == "MonitoringAgent"), None)
    assert monitor_log is not None, "MonitoringAgent should have logged an action."
    assert "redirect_off_topic" in monitor_log.message, \
        "Monitor should have logged the 'redirect_off_topic' decision."

def test_multilingual_response():
    """
    Tests if the UltraOrchestrator can respond in the user's language.
    """
    prompt = "Tôi muốn xây dựng một ứng dụng để theo dõi các mục tiêu thể dục của mình."
    final_state = run_workflow_with_prompt(prompt)

    # Check the assistant's response is in Vietnamese
    assistant_response = final_state.messages[-1].content
    # A simple check for a common Vietnamese word that should be in the response
    assert "giúp bạn" in assistant_response.lower() or "tính năng" in assistant_response.lower(), \
        "Assistant's response should be in Vietnamese."
        
def test_sufficient_request_creates_plan():
    """
    Tests if the UltraOrchestrator creates a plan when given a sufficiently detailed request.
    """
    prompt = (
        "I want to build a web app for tracking personal fitness goals. "
        "It needs user authentication (login/signup), a dashboard where users can log daily workouts (like 'Running, 3 miles'), "
        "and a progress page with charts showing workout frequency and distance over time. "
        "Please use React for the frontend and FastAPI for the backend."
    )
    final_state = run_workflow_with_prompt(prompt)
    
    # 1. Check the orchestrator's decision
    assert final_state.next_action == "create_plan", \
        "Orchestrator should have decided to create a plan."
        
    # 2. Check that tasks were created
    assert len(final_state.tasks) > 0, "A plan with at least one task should have been created."
    
    # 3. Check that the first task is a research task
    assert final_state.tasks[0].agent_type == AgentType.RESEARCH_AGENT, \
        "The first task in the plan should be for the Research Agent."
        
    # 4. Check the monitor's log
    monitor_log = next((log for log in reversed(final_state.logs) if log.source == "MonitoringAgent"), None)
    assert monitor_log is not None, "MonitoringAgent should have logged an action."
    assert "create_plan" in monitor_log.message, "Monitor should have logged the 'create_plan' decision."
    assert "task(s)" in monitor_log.message, "Monitor's log should mention the number of tasks created."

# To run these tests, use `pytest` in your terminal from the project root.
# Example: poetry run pytest tests/test_phase2_orchestration.py