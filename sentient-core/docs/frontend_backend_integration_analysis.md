# Frontend-Backend Integration Analysis

## Overview

This document provides a comprehensive analysis of the frontend-backend integration architecture for the Sentient Core multi-agent RAG system. The integration spans across all three builds, with each build introducing enhanced capabilities and more sophisticated communication patterns.

## Architecture Overview

### Technology Stack
- **Frontend**: Next.js 14 with React, TypeScript, Tailwind CSS
- **Backend**: FastAPI with Python, Pydantic for data validation
- **Communication**: REST APIs, Server-Sent Events (SSE), WebSocket connections
- **State Management**: React Context API, custom hooks, session persistence

## API Layer Architecture

### FastAPI Backend Structure

#### Main Application (`app/app.py`)
```python
# Core FastAPI application with CORS configuration
app = FastAPI(title="Sentient Core API")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Router registration with /api prefix
app.include_router(agents.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(core_services.router, prefix="/api")
app.include_router(api_endpoints.router, prefix="/api")
app.include_router(interactive_workflows.router, prefix="/api")
app.include_router(research.router, prefix="/api")
app.include_router(monitoring.router, prefix="/api")
app.include_router(implementation.router, prefix="/api")
```

#### API Router Structure

##### 1. Chat Router (`/api/chat/*`)
**Core Endpoints:**
- `POST /api/chat/message` - Process chat messages with optional image attachments
- `POST /api/chat/message/json` - JSON-only message processing (backward compatibility)
- `GET /api/chat/history` - Retrieve conversation history
- `DELETE /api/chat/history` - Clear chat history
- `GET /api/chat/sessions` - List all chat sessions
- `DELETE /api/chat/sessions/{session_id}` - Delete specific session
- `GET /api/chat/sessions/{session_id}/stats` - Get session statistics
- `POST /api/chat/confirmation` - Handle user confirmations
- `GET /api/chat/sessions/{session_id}/context` - Get conversation context
- `PUT /api/chat/sessions/{session_id}/context` - Update conversation context
- `GET /api/chat/sessions/{session_id}/confirmations` - Get pending confirmations
- `GET /api/chat/download/research/{filename}` - Download research artifacts
- `GET /api/chat/research/artifacts` - List research artifacts

**Key Features:**
- Session-based conversation management
- Multi-modal input support (text + images)
- Integration with `sentient_workflow_graph`
- Research artifact management
- User confirmation workflows

##### 2. Research Router (`/api/research/*`)
**Core Endpoints:**
- `POST /api/research/start` - Initiate research tasks

**Key Features:**
- Background research execution with `ResearchAgent`
- Real-time progress updates via SSE
- Research result persistence to file system
- Integration with memory service
- Support for multiple research modes (knowledge, deep, best_in_class)

##### 3. Agents Router (`/api/agents/*`)
**Core Endpoints:**
- `GET /api/agents/` - List available agent types
- `GET /api/agents/{agent_type}/info` - Get agent information
- `POST /api/agents/execute` - Execute agent tasks

**Agent Types:**
- `ultra_orchestrator` - Central control and task delegation
- `research_agent` - Knowledge acquisition and synthesis
- `architect_planner` - Planning and PRD generation
- `frontend_developer` - UI/UX development
- `backend_developer` - API and system architecture
- `coding_agent` - Code implementation
- `monitoring_agent` - System oversight
- `specialized_agent` - Custom domain-specific tasks

##### 4. Workflows Router (`/api/workflows/*`)
**Core Endpoints:**
- `GET /api/workflows/` - List available workflows
- `POST /api/workflows/{workflow_id}/execute` - Execute workflows
- `POST /api/workflows/tasks/{task_id}/execute` - Execute specific tasks
- `GET /api/workflows/tasks/status` - Get task status

**Workflow Types:**
- `intelligent` - Intelligent RAG with natural language orchestration
- `multi_agent` - Collaborative multi-agent system
- `legacy` - Standard orchestration workflow

##### 5. Core Services Router (`/api/core-services/*`)
**Memory Management:**
- `POST /api/core-services/memory/store` - Store memories
- `POST /api/core-services/memory/retrieve` - Retrieve memories
- `GET /api/core-services/memory/stats` - Memory statistics

**State Management:**
- `GET /api/core-services/state/agents` - Get agent states
- `GET /api/core-services/state/workflow/{workflow_id}` - Get workflow state
- `POST /api/core-services/state/agents/{agent_id}/update` - Update agent state
- `POST /api/core-services/state/workflow/{workflow_id}/update` - Update workflow state

**Search Services:**
- `POST /api/core-services/search/knowledge` - Knowledge search
- `GET /api/core-services/search/providers` - Search providers

##### 6. Interactive Workflows Router (`/api/interactive-workflows/*`)
**Enhanced workflow management for Build 2+**

##### 7. SSE Events Router (`/api/sse/*`)
**Real-time communication via Server-Sent Events**

## Frontend Architecture

### Next.js Application Structure

#### Main Application (`frontend/app/page.tsx`)
```typescript
// Multi-tab interface with workflow selection
<Tabs>
  <TabsTrigger value="orchestrator">Orchestrator</TabsTrigger>
  <TabsTrigger value="core-services">Core Services</TabsTrigger>
  <TabsTrigger value="interactive-workflows">Interactive Workflows</TabsTrigger>
  <TabsTrigger value="chat">Chat</TabsTrigger>
  <TabsTrigger value="tasks">Tasks</TabsTrigger>
  <TabsTrigger value="agents">Agents</TabsTrigger>
  <TabsTrigger value="llm">LLM Stream</TabsTrigger>
</Tabs>
```

#### Key Components

##### 1. Chat Interface (`components/chat-interface.tsx`)
**Features:**
- Real-time message display
- Multi-modal input (text + images)
- Integration with ChatService API client
- Message history management
- Research results integration
- Verbose feedback display

##### 2. Orchestrator Interface (`components/orchestrator-interface.tsx`)
**Features:**
- Advanced conversation management
- Agent state monitoring
- Workflow execution control
- Real-time status updates
- Confirmation handling

##### 3. Research Results (`components/research-results.tsx`)
**Features:**
- Research progress tracking
- Result visualization
- Artifact download capabilities
- Real-time updates via SSE

##### 4. Implementation Components
- `implementation-progress.tsx` - Track implementation progress
- `implementation-workflow.tsx` - Manage implementation workflows

### API Client Architecture

#### Core Services Client (`lib/api/core-services.ts`)
```typescript
export class CoreServicesClient {
  // Memory Management
  async storeMemory(request: MemoryStoreRequest)
  async retrieveMemory(request: MemoryRetrieveRequest)
  
  // State Management
  async getAgentStates()
  async getWorkflowState(workflowId: string)
  async updateAgentState(agentId: string, updates: Record<string, any>)
  
  // Search Services
  async searchKnowledge(request: SearchRequest)
}
```

#### Chat Service (`lib/api/chat-service.ts`)
```typescript
export const ChatService = {
  sendMessage: async (request: SendMessageRequest): Promise<Message>
  getChatHistory: async (workflow_mode?: string, task_id?: string): Promise<ChatHistory>
  clearChatHistory: async (workflow_mode?: string): Promise<{ success: boolean }>
}
```

#### Service Integration Pattern
```typescript
// Unified API exports
export { AgentService } from './agent-service';
export { WorkflowService } from './workflow-service';
export { InteractiveWorkflowService } from './interactive-workflow-service';
export { ChatService } from './chat-service';
export { CoreServicesClient, coreServicesClient } from './core-services';
```

## State Management Integration

### Frontend State Management

#### App Context (`lib/context/app-context.tsx`)
```typescript
interface AppContextType {
  activeWorkflow: string;
  setActiveWorkflow: (workflow: string) => void;
  resetSession: () => void;
  // Additional state management
}
```

#### Custom Hooks

##### 1. Orchestrator State (`lib/hooks/useOrchestratorState.ts`)
```typescript
export interface OrchestratorMessage {
  id: string;
  type: 'user' | 'orchestrator' | 'agent' | 'system' | 'confirmation' | 'artifact';
  content: string;
  timestamp: Date;
  metadata?: {
    agent_id?: string;
    workflow_id?: string;
    action_type?: string;
    requires_confirmation?: boolean;
    artifact_type?: 'research' | 'plan' | 'specification';
  };
}
```

##### 2. Artifact Manager (`lib/hooks/useArtifactManager.ts`)
```typescript
export interface Artifact {
  id: string;
  type: 'research' | 'plan' | 'specification' | 'code' | 'documentation';
  title: string;
  content?: string;
  url?: string;
  download_url?: string;
  metadata: {
    created_at: Date;
    updated_at: Date;
  };
}
```

### Backend State Management

#### Session Persistence
- Session-based conversation tracking
- Persistent storage in file system
- Memory service integration
- State synchronization across requests

#### Memory Layers
- **Layer 1**: Research documents (`memory/layer1_research_docs`)
- **Layer 2**: Planning documents (`memory/layer2_planning_docs`)
- **Conversation History**: Session-based storage
- **Knowledge Synthesis**: Long-term memory storage

## Real-Time Communication

### Server-Sent Events (SSE)

#### Research Progress Updates
```typescript
// Frontend SSE consumption
const eventSource = new EventSource(`/api/sse/research/${researchId}`);
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateResearchProgress(data);
};
```

#### Backend SSE Implementation
```python
# Real-time research updates
async def send_research_update(research_id: str, update_data: Dict[str, Any]):
    await send_sse_event(research_id, {
        "type": "research_update",
        "data": update_data,
        "timestamp": datetime.now().isoformat()
    })
```

### WebSocket Integration
- Workflow state synchronization
- Agent communication
- Real-time collaboration features

## Build-Specific Integration Features

### Build 1: Basic Chat Functionality
**Integration Points:**
- Simple chat API endpoints
- Basic message processing
- Session management
- File upload support

**Key Components:**
- `POST /api/chat/message` - Core messaging
- `GET /api/chat/history` - History retrieval
- Basic React chat interface
- Session persistence

### Build 2: Enhanced Multi-Agent System
**Integration Points:**
- Interactive workflow management
- Enhanced agent coordination
- Advanced state tracking
- Real-time progress monitoring

**Key Components:**
- Interactive workflow dashboard
- Agent state monitoring
- Enhanced confirmation workflows
- SSE-based real-time updates

### Build 3: Architect Planner + Tiered Memory
**Integration Points:**
- Planning workflow integration
- Tiered memory system
- PRD generation and storage
- Research-to-planning transitions

**Key Components:**
- `ArchitectPlannerAgent` integration
- Layer 2 memory persistence
- Planning state management
- Enhanced orchestration workflows

## Security and Error Handling

### CORS Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Error Handling Patterns

#### Backend Error Handling
```python
try:
    # API operation
    result = await process_request(request)
    return JSONResponse({"success": True, "data": result})
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Operation failed: {str(e)}")
```

#### Frontend Error Handling
```typescript
try {
  const response = await fetch(endpoint, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return await response.json();
} catch (error) {
  console.error('API Error:', error);
  throw error;
}
```

### Session Security
- Session ID validation
- Request sanitization
- File upload restrictions
- API rate limiting considerations

## Performance Considerations

### Frontend Optimizations
- Client-side rendering with Next.js
- Component lazy loading
- State management optimization
- API request caching

### Backend Optimizations
- Async/await patterns
- Background task execution
- Memory service caching
- File system optimization

### Communication Optimizations
- SSE for real-time updates
- Efficient JSON serialization
- Chunked response handling
- Connection pooling

## Testing and Validation

### API Testing
- End-to-end API tests
- Integration test suites
- Mock service implementations
- Error scenario testing

### Frontend Testing
- Component unit tests
- Integration testing
- User interaction testing
- Cross-browser compatibility

## Deployment Considerations

### Development Environment
- Frontend: `http://localhost:3000` (Next.js dev server)
- Backend: `http://localhost:8000` (FastAPI with uvicorn)
- Hot reload and development tools

### Production Deployment
- Environment variable configuration
- CORS policy adjustment
- SSL/TLS termination
- Load balancing considerations

## Future Enhancement Opportunities

### Real-Time Collaboration
- WebSocket-based real-time editing
- Multi-user session management
- Conflict resolution mechanisms

### Advanced State Management
- Redux/Zustand integration
- Persistent state hydration
- Optimistic UI updates

### Enhanced Security
- Authentication and authorization
- API key management
- Request validation middleware

### Performance Monitoring
- API response time tracking
- Frontend performance metrics
- Error rate monitoring
- User experience analytics

## Conclusion

The frontend-backend integration architecture provides a robust foundation for the Sentient Core multi-agent system. The modular design allows for incremental enhancement across builds while maintaining backward compatibility and ensuring scalable communication patterns. The combination of REST APIs, SSE, and WebSocket connections provides comprehensive real-time capabilities, while the structured state management ensures consistent user experiences across all interaction modes.