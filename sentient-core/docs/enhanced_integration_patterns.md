# Enhanced Integration Patterns

## Overview

This document details the advanced integration patterns implemented in the Sentient Core system, covering real-time communication, API orchestration, error handling strategies, performance optimization techniques, and cross-build feature integration.

## Real-Time Communication Patterns

### Server-Sent Events (SSE) Architecture

#### SSE Connection Management

**Core Components:**
- `SSEConnectionManager`: Manages active SSE connections
- `SSEEventHandler`: Processes and broadcasts events
- Connection lifecycle management with heartbeat monitoring

```python
class SSEConnectionManager:
    def __init__(self):
        self.connections: Dict[str, SSEConnection] = {}
        self.workflow_subscriptions: Dict[str, Set[str]] = {}
        self.research_subscriptions: Dict[str, Set[str]] = {}
        self.user_subscriptions: Dict[str, Set[str]] = {}
    
    async def create_connection(self, connection_id: str) -> SSEConnection:
        """Create new SSE connection with unique ID"""
        
    async def broadcast_to_workflow(self, message: dict, workflow_id: str):
        """Send message to all connections subscribed to workflow"""
        
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all active connections"""
```

#### Event Broadcasting Patterns

**Event Types:**
- `workflow_created`: New workflow initialization
- `step_started`: Workflow step execution begins
- `approval_requested`: User approval required
- `workflow_completed`: Workflow execution finished
- `workflow_error`: Error occurred during execution
- `research_update`: Research progress updates

**Broadcasting Strategy:**
```python
# Targeted broadcasting based on subscription type
await self.manager.broadcast_to_workflow(message, workflow_id)  # Workflow-specific
await self.manager.broadcast_to_research(message, research_id)  # Research-specific
await self.manager.broadcast_to_user(message, user_id)         # User-specific
await self.manager.broadcast_to_all(message)                   # Global broadcast
```

#### Connection Lifecycle

**Connection Establishment:**
1. Client initiates SSE connection to `/workflows` endpoint
2. Server creates unique connection ID
3. Connection registered in `SSEConnectionManager`
4. Initial connection message sent
5. Periodic heartbeat messages maintain connection

**Message Processing:**
```python
async def get_message_stream(self, connection_id: str):
    """Generate SSE message stream with heartbeat"""
    while connection_id in self.connections:
        try:
            # Send heartbeat every 30 seconds
            if time.time() - last_heartbeat > 30:
                yield f"data: {json.dumps({'type': 'heartbeat'})}

"
            
            # Process queued messages
            while not message_queue.empty():
                message = await message_queue.get()
                yield f"data: {json.dumps(message)}

"
                
        except Exception as e:
            await self.disconnect_connection(connection_id)
            break
```

### WebSocket Integration

**Workflow-Specific WebSockets:**
- Enhanced real-time updates for complex workflows
- Bidirectional communication for interactive features
- Integration with SSE for hybrid communication patterns

## API Orchestration Patterns

### FastAPI Router Architecture

#### Modular Router Structure

**Core Routers:**
```python
# Main application router registration
app.include_router(agents.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(core_services.router, prefix="/api")
app.include_router(research.router, prefix="/api")
app.include_router(monitoring.router, prefix="/api")
app.include_router(interactive_workflows.router, prefix="/api")
```

#### API Endpoint Patterns

**RESTful Design:**
- `GET /api/workflows/` - List available workflows
- `POST /api/workflows/{workflow_id}/execute` - Execute workflows
- `GET /api/workflows/tasks/status` - Get task status
- `POST /api/research/start` - Start research tasks
- `GET /api/chat/history` - Session-aware conversation history

**Request/Response Patterns:**
```python
@router.post("/start")
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    """Start research with background task execution"""
    try:
        research_id = str(uuid.uuid4())
        
        # Create research result entry
        research_result = ResearchResult(
            id=research_id,
            query=request.query,
            status="pending"
        )
        
        # Start background task
        background_tasks.add_task(
            execute_research_task,
            research_id,
            request
        )
        
        return JSONResponse({
            "success": True,
            "data": research_result.dict()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start research: {str(e)}"
        )
```

### Background Task Management

#### FastAPI Background Tasks

**Implementation Pattern:**
```python
from fastapi import BackgroundTasks

# Add background task to request
background_tasks.add_task(
    execute_research_task,
    research_id,
    request_parameters
)

# Background task execution
async def execute_research_task(research_id: str, request: ResearchRequest):
    """Execute long-running research task"""
    try:
        # Initialize research agent
        research_agent = ResearchAgent(llm_service)
        
        # Update progress periodically
        await update_research_status(
            research_id,
            "searching",
            progress_percentage,
            status_message
        )
        
        # Execute research steps
        for step in research_steps:
            result = await research_agent.execute_step(step)
            await send_real_time_update(research_id, result)
            
    except Exception as e:
        await handle_background_task_error(research_id, e)
```

#### Progress Tracking

**Real-time Progress Updates:**
```python
async def update_research_status(
    research_id: str,
    status: str,
    progress: float,
    message: str,
    logs: List[str] = None,
    results: Dict = None
):
    """Update research status with real-time broadcasting"""
    
    # Update in-memory storage
    research_storage[research_id].status = status
    research_storage[research_id].progress = progress
    
    # Broadcast via SSE
    await send_research_update(research_id, {
        "type": "progress_update",
        "status": status,
        "progress": progress,
        "message": message
    })
```

## Error Handling Strategies

### Backend Error Handling

#### Exception Hierarchy

**FastAPI Exception Patterns:**
```python
from fastapi import HTTPException

# Standard error response pattern
try:
    result = await process_request(request)
    return JSONResponse({
        "success": True,
        "data": result
    })
except ValidationError as e:
    raise HTTPException(
        status_code=422,
        detail=f"Validation error: {str(e)}"
    )
except ServiceUnavailableError as e:
    raise HTTPException(
        status_code=503,
        detail=f"Service unavailable: {str(e)}"
    )
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail=f"Internal server error: {str(e)}"
    )
```

#### Agent Error Handling

**BaseAgent Error Management:**
```python
class BaseAgent(ABC):
    async def handle_error(self, error: Exception, context: str = None):
        """Centralized agent error handling"""
        self.error_count += 1
        self.last_error = str(error)
        
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }
        
        # Log error
        await self.log_error(error_details)
        
        # Broadcast error event
        await self.broadcast_error_event(error_details)
        
        # Attempt recovery if possible
        if self.can_recover_from_error(error):
            await self.attempt_recovery(error, context)
```

#### Workflow Error Recovery

**Error Recovery Patterns:**
```python
async def handle_workflow_error(self, event_data: Dict[str, Any]):
    """Handle workflow error events with recovery attempts"""
    workflow_id = event_data.get("workflow_id")
    error_type = event_data.get("error_type")
    
    # Broadcast error to connected clients
    message = {
        "type": "workflow_error",
        "workflow_id": workflow_id,
        "data": event_data,
        "timestamp": datetime.now().isoformat(),
        "severity": "error"
    }
    await self.manager.broadcast_to_workflow(message, workflow_id)
    
    # Attempt automatic recovery for recoverable errors
    if error_type in RECOVERABLE_ERRORS:
        await self.attempt_workflow_recovery(workflow_id, event_data)
```

### Frontend Error Handling

#### API Client Error Management

**TypeScript Error Handling:**
```typescript
// Centralized API error handling
async function apiRequest<T>(endpoint: string, options: RequestInit): Promise<T> {
  try {
    const response = await fetch(endpoint, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData
      );
    }
    
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    
    // Network or parsing errors
    throw new APIError(
      `Network error: ${error.message}`,
      0,
      { originalError: error }
    );
  }
}

// Custom error class
class APIError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public details: any = {}
  ) {
    super(message);
    this.name = 'APIError';
  }
}
```

#### Error Boundary Implementation

**React Error Boundaries:**
```typescript
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  
  componentDidCatch(error, errorInfo) {
    // Log error to monitoring service
    console.error('Error caught by boundary:', error, errorInfo);
    
    // Send error report to backend
    this.reportError(error, errorInfo);
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallback error={this.state.error} />;
    }
    
    return this.props.children;
  }
}
```

## Performance Optimization Techniques

### Backend Optimizations

#### Async/Await Patterns

**Concurrent Processing:**
```python
# Parallel execution of independent tasks
async def process_multiple_requests(requests: List[Request]):
    """Process multiple requests concurrently"""
    tasks = [
        process_single_request(request)
        for request in requests
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results and exceptions
    successful_results = []
    errors = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append({"request_id": requests[i].id, "error": str(result)})
        else:
            successful_results.append(result)
    
    return successful_results, errors
```

#### Caching Strategies

**Memory-based Caching:**
```python
from functools import lru_cache
from typing import Dict, Any

class CacheManager:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.ttl_cache: Dict[str, float] = {}
    
    async def get_cached_result(self, key: str, ttl: int = 300):
        """Get cached result with TTL support"""
        if key in self.cache:
            if time.time() - self.ttl_cache.get(key, 0) < ttl:
                return self.cache[key]
            else:
                # Cache expired
                del self.cache[key]
                del self.ttl_cache[key]
        return None
    
    async def set_cached_result(self, key: str, value: Any):
        """Set cached result with timestamp"""
        self.cache[key] = value
        self.ttl_cache[key] = time.time()

# Usage in API endpoints
@lru_cache(maxsize=128)
def get_agent_configuration(agent_type: str) -> Dict[str, Any]:
    """Cached agent configuration lookup"""
    return load_agent_config(agent_type)
```

#### Database Optimization

**Connection Pooling and Batch Operations:**
```python
class DatabaseOptimizer:
    def __init__(self, pool_size: int = 10):
        self.connection_pool = create_pool(size=pool_size)
    
    async def batch_insert(self, table: str, records: List[Dict]):
        """Optimized batch insert operation"""
        async with self.connection_pool.acquire() as conn:
            # Use batch insert for better performance
            await conn.executemany(
                f"INSERT INTO {table} VALUES (...)",
                records
            )
    
    async def bulk_update_status(self, updates: List[Tuple[str, str]]):
        """Bulk status updates with single transaction"""
        async with self.connection_pool.acquire() as conn:
            async with conn.transaction():
                for entity_id, status in updates:
                    await conn.execute(
                        "UPDATE entities SET status = $1 WHERE id = $2",
                        status, entity_id
                    )
```

### Frontend Optimizations

#### Component Optimization

**React Performance Patterns:**
```typescript
// Memoized components for expensive renders
const ExpensiveComponent = React.memo(({ data, onUpdate }) => {
  const processedData = useMemo(() => {
    return expensiveDataProcessing(data);
  }, [data]);
  
  const handleUpdate = useCallback((newValue) => {
    onUpdate(newValue);
  }, [onUpdate]);
  
  return (
    <div>
      {processedData.map(item => (
        <Item key={item.id} data={item} onUpdate={handleUpdate} />
      ))}
    </div>
  );
});

// Virtualized lists for large datasets
const VirtualizedList = ({ items }) => {
  return (
    <FixedSizeList
      height={600}
      itemCount={items.length}
      itemSize={50}
      itemData={items}
    >
      {({ index, style, data }) => (
        <div style={style}>
          <ListItem data={data[index]} />
        </div>
      )}
    </FixedSizeList>
  );
};
```

#### State Management Optimization

**Efficient State Updates:**
```typescript
// Context optimization with selective updates
const StateContext = createContext();

const StateProvider = ({ children }) => {
  const [state, setState] = useState(initialState);
  
  // Memoized state selectors
  const selectors = useMemo(() => ({
    getWorkflowById: (id: string) => state.workflows[id],
    getActiveWorkflows: () => Object.values(state.workflows)
      .filter(w => w.status === 'active'),
    getWorkflowProgress: (id: string) => state.workflows[id]?.progress || 0
  }), [state]);
  
  // Optimized update functions
  const updateWorkflowStatus = useCallback((id: string, status: string) => {
    setState(prev => ({
      ...prev,
      workflows: {
        ...prev.workflows,
        [id]: { ...prev.workflows[id], status }
      }
    }));
  }, []);
  
  return (
    <StateContext.Provider value={{ state, selectors, updateWorkflowStatus }}>
      {children}
    </StateContext.Provider>
  );
};
```

## Cross-Build Feature Integration

### Build Progression Architecture

#### Build 1: Basic Chat Functionality

**Core Integration Points:**
- FastAPI router integration
- Session management
- Basic state persistence
- Simple error handling

**Dependencies:**
```python
# Build 1 core dependencies
from fastapi import FastAPI, Depends, HTTPException
from core.models import AppState, Message
from core.services.session_service import SessionPersistenceService
from core.state.enhanced_state_manager import EnhancedStateManager
```

#### Build 2: Ultra Orchestrator + Research Agent

**Enhanced Integration:**
- Multi-agent orchestration
- Research task delegation
- Advanced state management
- Real-time progress tracking

**Integration Patterns:**
```python
# Build 2 orchestration integration
class UltraOrchestrator:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.state_manager = EnhancedStateManager()
        self.sse_manager = SSEConnectionManager()
    
    async def delegate_research_task(self, query: str, session_id: str):
        """Delegate research to specialized agent"""
        # Create research task
        research_id = await self.research_agent.start_research(query)
        
        # Update orchestrator state
        await self.state_manager.update_state({
            f"sessions.{session_id}.active_research": research_id,
            f"research.{research_id}.status": "delegated"
        })
        
        # Broadcast delegation event
        await self.sse_manager.broadcast_to_user({
            "type": "research_delegated",
            "research_id": research_id,
            "query": query
        }, session_id)
```

#### Build 3+: Advanced Multi-Agent System

**Planned Integration Enhancements:**
- Cross-agent communication protocols
- Distributed task execution
- Advanced workflow orchestration
- Multi-user collaboration support

### Dependency Management

#### Service Dependencies

**Dependency Injection Pattern:**
```python
from typing import Protocol

class MemoryServiceProtocol(Protocol):
    async def store_memory(self, memory_type: MemoryType, content: Dict) -> str:
        ...
    
    async def retrieve_memory(self, memory_id: str) -> Optional[Dict]:
        ...

class StateManagerProtocol(Protocol):
    async def get_state(self, path: str) -> Any:
        ...
    
    async def update_state(self, updates: Dict[str, Any]) -> None:
        ...

# Service factory with dependency injection
class ServiceFactory:
    def __init__(self):
        self._memory_service: Optional[MemoryServiceProtocol] = None
        self._state_manager: Optional[StateManagerProtocol] = None
    
    def get_memory_service(self) -> MemoryServiceProtocol:
        if not self._memory_service:
            self._memory_service = MemoryService()
        return self._memory_service
    
    def get_state_manager(self) -> StateManagerProtocol:
        if not self._state_manager:
            self._state_manager = EnhancedStateManager()
        return self._state_manager

# Global service factory
service_factory = ServiceFactory()
```

#### Configuration Management

**Environment-based Configuration:**
```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "localhost"
    api_port: int = 8000
    api_prefix: str = "/api"
    
    # Database Configuration
    database_url: str = "sqlite:///./sentient_core.db"
    
    # Cache Configuration
    cache_ttl: int = 300
    cache_max_size: int = 1000
    
    # SSE Configuration
    sse_heartbeat_interval: int = 30
    sse_connection_timeout: int = 300
    
    # Security Configuration
    cors_origins: List[str] = ["*"]
    api_key_required: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
```

## Security and Middleware

### CORS Configuration

**Production-Ready CORS:**
```python
from fastapi.middleware.cors import CORSMiddleware

# Development configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Production configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sentientcore.app",
        "https://app.sentientcore.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### Request Validation

**Input Sanitization:**
```python
from pydantic import BaseModel, validator
from typing import Optional
import re

class SecureRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    
    @validator('query')
    def validate_query(cls, v):
        # Remove potentially harmful characters
        if not v or len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        
        # Limit query length
        if len(v) > 1000:
            raise ValueError("Query too long")
        
        # Basic XSS prevention
        dangerous_patterns = [r'<script', r'javascript:', r'on\w+=']
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Invalid characters in query")
        
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9-_]{1,50}$', v):
            raise ValueError("Invalid session ID format")
        return v
```

### Rate Limiting

**API Rate Limiting:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply rate limiting to endpoints
@router.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """Chat endpoint with rate limiting"""
    return await process_chat_request(chat_request)

@router.post("/research/start")
@limiter.limit("5/minute")
async def start_research(request: Request, research_request: ResearchRequest):
    """Research endpoint with stricter rate limiting"""
    return await start_research_task(research_request)
```

## Monitoring and Analytics

### Performance Monitoring

**Request Timing Middleware:**
```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 1.0:  # Log requests taking more than 1 second
        logger.warning(
            f"Slow request: {request.method} {request.url.path} "
            f"took {process_time:.2f}s"
        )
    
    return response
```

### Error Tracking

**Centralized Error Logging:**
```python
import logging
from datetime import datetime

class ErrorTracker:
    def __init__(self):
        self.error_counts = {}
        self.recent_errors = []
    
    async def track_error(self, error: Exception, context: Dict[str, Any]):
        """Track error occurrence with context"""
        error_key = f"{type(error).__name__}:{str(error)}"
        
        # Update error counts
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Store recent error details
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "count": self.error_counts[error_key]
        }
        
        self.recent_errors.append(error_details)
        
        # Keep only last 100 errors
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)
        
        # Log error
        logging.error(f"Error tracked: {error_details}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts),
            "error_counts": self.error_counts,
            "recent_errors": self.recent_errors[-10:]  # Last 10 errors
        }

# Global error tracker
error_tracker = ErrorTracker()
```

## Conclusion

The enhanced integration patterns documented here provide a comprehensive foundation for building scalable, maintainable, and performant features in the Sentient Core system. These patterns emphasize:

1. **Real-time Communication**: Robust SSE implementation with connection management and event broadcasting
2. **API Orchestration**: Modular router architecture with background task support
3. **Error Handling**: Comprehensive error management at both backend and frontend levels
4. **Performance Optimization**: Caching, async processing, and efficient state management
5. **Cross-Build Integration**: Progressive feature enhancement with dependency management
6. **Security**: Input validation, rate limiting, and secure middleware implementation
7. **Monitoring**: Performance tracking and error analytics for system health

These patterns should be consistently applied across all builds to ensure system coherence and maintainability as the platform evolves.