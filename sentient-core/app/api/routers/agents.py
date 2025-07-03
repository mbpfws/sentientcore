from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models import AgentType, EnhancedTask, TaskStatus
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.agents.research_agent import ResearchAgent
from core.agents.architect_planner_agent import ArchitectPlannerAgent
from core.agents.frontend_developer_agent import FrontendDeveloperAgent
from core.agents.backend_developer_agent import BackendDeveloperAgent
from core.agents.coding_agent import CodingAgent
from core.agents.monitoring_agent import MonitoringAgent

router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/")
async def list_agents():
    """List all available agent types"""
    return [
        {"type": agent_type.value, "name": agent_type.name} 
        for agent_type in AgentType
    ]

@router.get("/{agent_type}/info")
async def get_agent_info(agent_type: str):
    """Get detailed information about a specific agent"""
    try:
        # Convert string to enum
        agent_enum = AgentType(agent_type)
        
        # Agent descriptions
        agent_descriptions = {
            AgentType.ULTRA_ORCHESTRATOR: "Central control unit managing user interaction and task delegation",
            AgentType.RESEARCH_AGENT: "Handles comprehensive research, knowledge acquisition, and synthesis",
            AgentType.ARCHITECT_PLANNER: "Creates plans and synthesizes knowledge into actionable development plans",
            AgentType.FRONTEND_DEVELOPER: "Specializes in front-end design and development",
            AgentType.BACKEND_DEVELOPER: "Handles back-end architecture and development",
            AgentType.CODING_AGENT: "Executor agent implementing specific code tasks",
            AgentType.MONITORING_AGENT: "Provides oversight of workflows and tracks progression",
            AgentType.SPECIALIZED_AGENT: "Custom agent for specialized tasks"
        }
        
        return {
            "type": agent_enum.value,
            "name": agent_enum.name,
            "description": agent_descriptions.get(agent_enum, "No description available"),
            "capabilities": get_agent_capabilities(agent_enum)
        }
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

def get_agent_capabilities(agent_type: AgentType) -> List[str]:
    """Get list of capabilities for a specific agent type"""
    capabilities = {
        AgentType.ULTRA_ORCHESTRATOR: [
            "Conversation management",
            "Task orchestration",
            "System control",
            "State management"
        ],
        AgentType.RESEARCH_AGENT: [
            "Knowledge research",
            "Deep research",
            "Best-in-class evaluation",
            "Report generation"
        ],
        AgentType.ARCHITECT_PLANNER: [
            "High-level planning",
            "Product requirement creation",
            "Technical specification development",
            "Action plan generation"
        ],
        AgentType.FRONTEND_DEVELOPER: [
            "UI/UX design",
            "Frontend implementation",
            "Component development",
            "Responsive design"
        ],
        AgentType.BACKEND_DEVELOPER: [
            "API design",
            "Database architecture",
            "Backend implementation",
            "System integration"
        ],
        AgentType.CODING_AGENT: [
            "Code implementation",
            "Bug fixing",
            "Testing",
            "Code optimization"
        ],
        AgentType.MONITORING_AGENT: [
            "Process monitoring",
            "Status reporting",
            "Alert generation",
            "System oversight"
        ],
        AgentType.SPECIALIZED_AGENT: [
            "Custom capabilities",
            "Specialized processing",
            "Domain-specific functions"
        ]
    }
    
    return capabilities.get(agent_type, ["No capabilities defined"])
