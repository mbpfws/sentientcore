"""Orchestration Module

Provides workflow orchestration capabilities for different execution modes
including intelligent RAG, multi-agent RAG, and legacy orchestration.
"""

from .workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowMode,
    ResearchMode,
    get_workflow_orchestrator,
    initialize_workflow_orchestrator,
    shutdown_workflow_orchestrator
)

__all__ = [
    'WorkflowOrchestrator',
    'WorkflowMode',
    'ResearchMode',
    'get_workflow_orchestrator',
    'initialize_workflow_orchestrator',
    'shutdown_workflow_orchestrator'
]