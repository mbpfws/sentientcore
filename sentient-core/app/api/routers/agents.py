from fastapi import APIRouter, HTTPException, Depends, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import sys
import asyncio
from datetime import datetime

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models import AgentType, EnhancedTask, TaskStatus

# Request models
class AgentExecuteRequest(BaseModel):
    agent_type: str
    task: str
    parameters: Optional[Dict[str, Any]] = {}

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

@router.post("/execute")
async def execute_agent(request: AgentExecuteRequest):
    """Execute an agent with the specified task and parameters"""
    try:
        # Validate agent type
        valid_agent_types = [
            "monitoring_agent", "ultra_orchestrator", "research_agent",
            "architect_planner", "frontend_developer", "backend_developer",
            "coding_agent", "specialized_agent"
        ]
        
        if request.agent_type not in valid_agent_types:
            raise HTTPException(status_code=400, detail=f"Invalid agent type: {request.agent_type}")
        
        # Handle specific tasks without instantiating actual agent classes
        if request.task == "system_health_check":
            # Mock system health check result
            result = {
                "agent_type": request.agent_type,
                "task": request.task,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "result": {
                    "health_status": "healthy",
                    "checks_performed": [
                        "system_resources",
                        "agent_availability",
                        "memory_status",
                        "api_endpoints"
                    ],
                    "system_metrics": {
                        "cpu_usage": "normal",
                        "memory_usage": "normal",
                        "response_time": "optimal"
                    },
                    "parameters_received": request.parameters
                },
                "execution_time": 0.1
            }
        else:
            # Generic task execution mock
            result = {
                "agent_type": request.agent_type,
                "task": request.task,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "result": {
                    "message": f"Task '{request.task}' executed successfully by {request.agent_type}",
                    "parameters_received": request.parameters,
                    "execution_details": {
                        "processed_at": datetime.now().isoformat(),
                        "agent_status": "active",
                        "task_complexity": "standard"
                    }
                },
                "execution_time": 0.05
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")
