# 03 - Agent Framework Enhancement

## Overview

This phase establishes a robust agent framework that provides the foundation for all specialized agents. We'll create base classes, communication protocols, and shared utilities that ensure consistent behavior across the multi-agent system.

## Current Agent Analysis

### Existing Agents
- `ultra_orchestrator.py` - Basic orchestration logic
- `research_agent.py` - Research capabilities
- `architect_planner_agent.py` - Planning functionality
- `frontend_developer_agent.py` - Frontend development
- `backend_developer_agent.py` - Backend development
- `coding_agent.py` - Code execution
- `monitoring_agent.py` - System monitoring

### Framework Requirements
- Standardized agent interface
- Inter-agent communication protocol
- Shared state management
- Error handling and recovery
- Logging and monitoring integration

## Implementation Tasks

### Task 3.1: Base Agent Framework

**File**: `core/agents/base_agent.py`

**Base Agent Class**:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum

class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"

class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        self.current_task = None
        self.capabilities = []
        self.memory_service = None
        self.state_service = None
        self.llm_service = None
        
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results"""
        pass
    
    @abstractmethod
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Determine if agent can handle the given task"""
        pass
    
    async def update_status(self, status: AgentStatus, details: str = None):
        """Update agent status and notify monitoring"""
        self.status = status
        await self.state_service.update_agent_state(self.agent_id, {
            'status': status.value,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def log_activity(self, activity: str, level: str = 'info'):
        """Log agent activity for monitoring"""
        await self.memory_service.store_knowledge(
            'conversation_history',
            {
                'agent_id': self.agent_id,
                'activity': activity,
                'level': level,
                'timestamp': datetime.utcnow().isoformat()
            },
            {'type': 'agent_log'}
        )
```

### Task 3.2: Agent Communication Protocol

**File**: `core/agents/communication.py`

**Message System**:
```python
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    COLLABORATION_REQUEST = "collaboration_request"
    ERROR_NOTIFICATION = "error_notification"
    SYSTEM_COMMAND = "system_command"

@dataclass
class AgentMessage:
    sender_id: str
    recipient_id: str
    message_type: MessageType
    content: Dict[str, Any]
    correlation_id: Optional[str] = None
    timestamp: Optional[str] = None
    priority: int = 1  # 1=low, 5=high

class MessageBus:
    def __init__(self):
        self.subscribers = {}
        self.message_queue = []
        
    async def send_message(self, message: AgentMessage):
        """Send message to specific agent"""
        pass
    
    async def broadcast_message(self, message: AgentMessage, agent_types: List[str]):
        """Broadcast message to multiple agent types"""
        pass
    
    async def subscribe(self, agent_id: str, message_types: List[MessageType]):
        """Subscribe agent to specific message types"""
        pass
```

### Task 3.3: Agent Registry and Discovery

**File**: `core/agents/registry.py`

**Agent Registry**:
```python
class AgentRegistry:
    def __init__(self):
        self.agents = {}
        self.agent_types = {}
        self.capabilities_map = {}
    
    async def register_agent(self, agent: BaseAgent):
        """Register new agent in the system"""
        self.agents[agent.agent_id] = agent
        agent_type = agent.__class__.__name__
        
        if agent_type not in self.agent_types:
            self.agent_types[agent_type] = []
        self.agent_types[agent_type].append(agent.agent_id)
        
        # Map capabilities
        for capability in agent.capabilities:
            if capability not in self.capabilities_map:
                self.capabilities_map[capability] = []
            self.capabilities_map[capability].append(agent.agent_id)
    
    async def find_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Find agents that have specific capability"""
        agent_ids = self.capabilities_map.get(capability, [])
        return [self.agents[agent_id] for agent_id in agent_ids]
    
    async def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    async def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get all agents of specific type"""
        agent_ids = self.agent_types.get(agent_type, [])
        return [self.agents[agent_id] for agent_id in agent_ids]
```

### Task 3.4: Enhanced Agent Models

**File**: `core/models.py` (Enhancement)

**Agent Data Models**:
```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class AgentCapability(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = {}

class AgentProfile(BaseModel):
    agent_id: str
    name: str
    description: str
    agent_type: str
    capabilities: List[AgentCapability]
    status: str
    created_at: datetime
    last_active: datetime

class TaskDefinition(BaseModel):
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = []
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: datetime
    deadline: Optional[datetime] = None

class WorkflowState(BaseModel):
    workflow_id: str
    name: str
    description: str
    current_phase: str
    tasks: List[TaskDefinition]
    agent_assignments: Dict[str, str]
    progress: float = 0.0
    status: str = "active"
    created_at: datetime
    updated_at: datetime
```

### Task 3.5: Agent Factory and Lifecycle Management

**File**: `core/agents/factory.py`

**Agent Factory**:
```python
class AgentFactory:
    def __init__(self, registry: AgentRegistry, services: Dict[str, Any]):
        self.registry = registry
        self.services = services
        self.agent_configs = {}
    
    async def create_agent(self, agent_type: str, config: Dict[str, Any]) -> BaseAgent:
        """Create and initialize new agent"""
        agent_class = self._get_agent_class(agent_type)
        agent = agent_class(**config)
        
        # Inject services
        agent.memory_service = self.services['memory']
        agent.state_service = self.services['state']
        agent.llm_service = self.services['llm']
        
        # Register agent
        await self.registry.register_agent(agent)
        
        return agent
    
    async def destroy_agent(self, agent_id: str):
        """Safely destroy agent and cleanup resources"""
        agent = await self.registry.get_agent_by_id(agent_id)
        if agent:
            await agent.cleanup()
            await self.registry.unregister_agent(agent_id)
    
    def _get_agent_class(self, agent_type: str):
        """Get agent class by type name"""
        agent_classes = {
            'UltraOrchestrator': UltraOrchestratorAgent,
            'Research': ResearchAgent,
            'ArchitectPlanner': ArchitectPlannerAgent,
            'FrontendDeveloper': FrontendDeveloperAgent,
            'BackendDeveloper': BackendDeveloperAgent,
            'Coding': CodingAgent,
            'Monitoring': MonitoringAgent
        }
        return agent_classes.get(agent_type)
```

### Task 3.6: Error Handling and Recovery

**File**: `core/agents/error_handling.py`

**Error Management**:
```python
class AgentError(Exception):
    def __init__(self, agent_id: str, error_type: str, message: str, context: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.error_type = error_type
        self.message = message
        self.context = context or {}
        super().__init__(f"Agent {agent_id}: {message}")

class ErrorRecoveryManager:
    def __init__(self, registry: AgentRegistry, state_service):
        self.registry = registry
        self.state_service = state_service
        self.recovery_strategies = {}
    
    async def handle_agent_error(self, error: AgentError):
        """Handle agent error with appropriate recovery strategy"""
        strategy = self.recovery_strategies.get(error.error_type, self._default_recovery)
        await strategy(error)
    
    async def _default_recovery(self, error: AgentError):
        """Default error recovery strategy"""
        agent = await self.registry.get_agent_by_id(error.agent_id)
        if agent:
            await agent.update_status(AgentStatus.ERROR, error.message)
            # Attempt to restart agent
            await self._restart_agent(agent)
    
    async def _restart_agent(self, agent: BaseAgent):
        """Restart agent after error"""
        try:
            await agent.reset()
            await agent.update_status(AgentStatus.IDLE)
        except Exception as e:
            # If restart fails, mark agent as failed
            await agent.update_status(AgentStatus.ERROR, f"Restart failed: {str(e)}")
```

## Backend API Integration

### Task 3.7: Agent Management Endpoints

**File**: `app/api/agents.py`

**Agent API Endpoints**:
```python
from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter(prefix="/api/agents", tags=["agents"])

@router.get("/", response_model=List[AgentProfile])
async def list_agents():
    """Get list of all registered agents"""
    pass

@router.get("/{agent_id}", response_model=AgentProfile)
async def get_agent(agent_id: str):
    """Get specific agent details"""
    pass

@router.post("/create")
async def create_agent(agent_type: str, config: Dict[str, Any]):
    """Create new agent instance"""
    pass

@router.post("/{agent_id}/task")
async def assign_task(agent_id: str, task: TaskDefinition):
    """Assign task to specific agent"""
    pass

@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get current agent status"""
    pass

@router.post("/{agent_id}/message")
async def send_message(agent_id: str, message: AgentMessage):
    """Send message to agent"""
    pass
```

## Frontend Integration

### Task 3.8: Agent Management Components

**File**: `frontend/components/agent-dashboard.tsx`

**Agent Dashboard Component**:
```typescript
import React, { useState, useEffect } from 'react';
import { AgentProfile, AgentStatus } from '../lib/api/types';

interface AgentDashboardProps {
  onAgentSelect: (agentId: string) => void;
}

export const AgentDashboard: React.FC<AgentDashboardProps> = ({ onAgentSelect }) => {
  const [agents, setAgents] = useState<AgentProfile[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  
  useEffect(() => {
    fetchAgents();
    const interval = setInterval(fetchAgents, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);
  
  const fetchAgents = async () => {
    // Fetch agent list from API
  };
  
  const getStatusColor = (status: string) => {
    const colors = {
      idle: 'bg-gray-500',
      thinking: 'bg-yellow-500',
      working: 'bg-blue-500',
      waiting: 'bg-orange-500',
      error: 'bg-red-500',
      completed: 'bg-green-500'
    };
    return colors[status] || 'bg-gray-500';
  };
  
  return (
    <div className="agent-dashboard">
      <h2 className="text-xl font-bold mb-4">Agent Status</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agents.map(agent => (
          <div 
            key={agent.agent_id}
            className={`p-4 border rounded-lg cursor-pointer hover:shadow-lg ${
              selectedAgent === agent.agent_id ? 'border-blue-500' : 'border-gray-200'
            }`}
            onClick={() => {
              setSelectedAgent(agent.agent_id);
              onAgentSelect(agent.agent_id);
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold">{agent.name}</h3>
              <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status)}`}></div>
            </div>
            <p className="text-sm text-gray-600 mb-2">{agent.description}</p>
            <div className="text-xs text-gray-500">
              Type: {agent.agent_type}
            </div>
            <div className="text-xs text-gray-500">
              Capabilities: {agent.capabilities.length}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```

### Task 3.9: Agent Communication Interface

**File**: `frontend/components/agent-communication.tsx`

**Message Interface**:
```typescript
export const AgentCommunication: React.FC<{ agentId: string }> = ({ agentId }) => {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  
  const sendMessage = async () => {
    // Send message to agent via API
  };
  
  return (
    <div className="agent-communication">
      <div className="messages-container h-64 overflow-y-auto border rounded p-4 mb-4">
        {messages.map(message => (
          <div key={message.correlation_id} className="message mb-2">
            <div className="text-sm text-gray-500">
              {message.sender_id} → {message.recipient_id}
            </div>
            <div className="text-sm">{JSON.stringify(message.content, null, 2)}</div>
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          className="flex-1 border rounded px-3 py-2"
          placeholder="Send message to agent..."
        />
        <button
          onClick={sendMessage}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Send
        </button>
      </div>
    </div>
  );
};
```

## Testing Strategy

### Task 3.10: Framework Testing

**Unit Tests**:
```python
# test_base_agent.py
class TestBaseAgent:
    async def test_agent_initialization(self):
        pass
    
    async def test_status_updates(self):
        pass
    
    async def test_message_handling(self):
        pass

# test_agent_registry.py
class TestAgentRegistry:
    async def test_agent_registration(self):
        pass
    
    async def test_capability_discovery(self):
        pass
    
    async def test_agent_lookup(self):
        pass
```

**Integration Tests**:
- Agent creation and registration
- Inter-agent communication
- Error handling and recovery
- Frontend-backend agent management

## Validation Criteria

### Backend Validation
- [ ] Base agent framework functional
- [ ] Agent registry manages agents correctly
- [ ] Message bus handles communication
- [ ] Error recovery mechanisms work
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] Agent dashboard displays real-time status
- [ ] Agent selection and interaction works
- [ ] Message interface functional
- [ ] Status updates reflect in UI

### Integration Validation
- [ ] Agents can be created and destroyed
- [ ] Inter-agent messages delivered correctly
- [ ] Status changes propagate to frontend
- [ ] Error states handled gracefully

## Human Testing Scenarios

1. **Agent Creation Test**: Create different agent types and verify registration
2. **Status Monitoring Test**: Monitor agent status changes in real-time
3. **Message Passing Test**: Send messages between agents and verify delivery
4. **Error Recovery Test**: Trigger agent errors and verify recovery
5. **Dashboard Interaction Test**: Use frontend to manage agents

## Next Steps

After successful validation of the agent framework, proceed to **04-ultra-orchestrator-implementation.md** for implementing the enhanced Ultra Orchestrator Agent that will coordinate all other agents.

---

**Dependencies**: This phase requires the core services from Phase 2 to be functional and builds the foundation for all subsequent agent implementations.