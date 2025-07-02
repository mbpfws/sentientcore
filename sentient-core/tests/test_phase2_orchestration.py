"""
Test suite for the Phase 2 implementation of the Sentient Agentic Framework.
Validates the core decision-making loop between the UltraOrchestrator and MonitoringAgent.
"""

import pytest
import os
from dotenv import load_dotenv

# Load environment variables for local testing
# The path resolution for imports is handled by tests/conftest.py
load_dotenv()

from core.models import AppState, Message, AgentType
from graphs.sentient_workflow_graph import app as sentient_workflow_app
from typing import Any

def run_workflow_with_prompt(prompt: str) -> AppState:
    """Helper function to run the workflow with a single user prompt."""
    initial_state = AppState(
        messages=[Message(sender="user", content=prompt)]
    )
    # Pass the initial state object directly to satisfy the type checker.
    # The graph will return a dictionary representing the final state.
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
    Tests if the UltraOrchestrator can respond in the user's language and
    correctly identifies the language in its decision.
    """
    prompt = "Tôi muốn xây dựng một ứng dụng để theo dõi các mục tiêu thể dục của mình."
    final_state = run_workflow_with_prompt(prompt)

    # 1. Check the orchestrator's decision for the detected language
    orchestrator_decision = final_state.orchestrator_decision
    assert orchestrator_decision is not None, "Orchestrator decision should not be None."
    assert orchestrator_decision.get("language_detected") == "vi", \
        "Orchestrator should have detected Vietnamese ('vi')."

    # 2. Check the assistant's response is in Vietnamese (optional, but good to have)
    assistant_response = final_state.messages[-1].content
    # A simple check for a common Vietnamese word that should be in the response
    assert "bạn" in assistant_response.lower() or "tôi" in assistant_response.lower(), \
        "Assistant's response should contain common Vietnamese words."
        
def test_sufficient_request_creates_plan_after_clarification():
    """
    Tests if the UltraOrchestrator creates a plan after a two-turn conversation
    where it first requests clarification and then receives sufficient details.
    """
    # Turn 1: Initial, slightly vague prompt
    initial_prompt = "I want to build a web app for tracking personal fitness goals using React and FastAPI."
    
    state_after_turn_1 = run_workflow_with_prompt(initial_prompt)
    
    # Assert that the orchestrator asks for clarification first
    assert state_after_turn_1.next_action == "request_clarification"
    
    # Turn 2: Provide the clarifying details
    clarification_prompt = (
        "It needs user authentication (login/signup), a dashboard where users can log daily workouts "
        "(like 'Running, 3 miles'), and a progress page with charts showing workout frequency and distance over time."
    )
    
    # Continue the conversation
    state_after_turn_1.messages.append(Message(sender="user", content=clarification_prompt))
    
    final_state_dict = sentient_workflow_app.invoke(state_after_turn_1)
    final_state = AppState(**final_state_dict)

    # Assert that the orchestrator now decides to create a plan
    assert final_state.next_action == "create_plan", \
        f"Orchestrator should have decided to create a plan after clarification. Got '{final_state.next_action}' instead."
        
    # Check that tasks were created
    assert len(final_state.tasks) > 0, "A plan with at least one task should have been created."
    
    # Check that the first task is a research task
    assert final_state.tasks[0].agent_type == AgentType.RESEARCH_AGENT, \
        "The first task in the plan should be for the Research Agent."
        
    # Check the monitor's log
    monitor_log = next((log for log in reversed(final_state.logs) if log.source == "MonitoringAgent"), None)
    assert monitor_log is not None, "MonitoringAgent should have logged an action."
    assert "create_plan" in monitor_log.message, "Monitor should have logged the 'create_plan' decision."
    assert "task(s)" in monitor_log.message, "Monitor's log should mention the number of tasks created."

# To run these tests, use `pytest` in your terminal from the project root.
# Example: poetry run pytest sentient-core/tests/test_phase2_orchestration.py 