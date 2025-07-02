# 02 - Core Services Enhancement

## Overview

This phase focuses on enhancing and expanding the core services that form the foundation of the multi-agent system. We'll improve existing services and add new essential services for robust agent orchestration.

## Current Services Analysis

### Existing Services
1. **LLM Service** (`core/services/llm_service.py`) - Basic LLM integration
2. **E2B Service** (`core/services/e2b_service.py`) - Code execution environment

### Required Enhancements
1. **Enhanced LLM Service** - Multi-provider support with fallbacks
2. **Memory Service** - 4-layer hierarchical knowledge base
3. **State Management Service** - Agent and workflow state tracking
4. **Search Service** - External search integration (Tavily/Exa)

## Implementation Tasks

### Task 2.1: Enhanced LLM Service

**File**: `core/services/llm_service.py`

**Enhancements**:
```python
class EnhancedLLMService:
    def __init__(self):
        self.providers = {
            'groq': GroqProvider(),
            'gemini': GeminiProvider(),
            'openai': OpenAIProvider()  # Fallback
        }
        self.fallback_chain = ['groq', 'gemini', 'openai']
    
    async def generate_with_fallback(self, prompt: str, model_preference: str = 'groq'):
        """Generate response with automatic fallback on failure"""
        pass
    
    async def stream_response(self, prompt: str, callback: callable):
        """Stream response for real-time UI updates"""
        pass
```

**Features to Implement**:
- Multi-provider support (Groq, Gemini, OpenAI)
- Automatic fallback mechanisms
- Streaming responses for real-time UI
- Token usage tracking and optimization
- Model-specific prompt optimization
- Rate limiting and retry logic

### Task 2.2: Memory Management Service

**File**: `core/services/memory_service.py`

**4-Layer Architecture**:
```python
class MemoryService:
    def __init__(self):
        self.layers = {
            'knowledge_synthesis': KnowledgeLayer(),
            'conversation_history': ConversationLayer(),
            'codebase_knowledge': CodebaseLayer(),
            'stack_dependencies': StackLayer()
        }
    
    async def store_knowledge(self, layer: str, data: dict, metadata: dict):
        """Store information in specified memory layer"""
        pass
    
    async def retrieve_relevant(self, query: str, layers: list = None):
        """Retrieve relevant information across layers"""
        pass
```

**Implementation Details**:
- **Layer 1 - Knowledge Synthesis**: Research findings, technical reports
- **Layer 2 - Conversation History**: User interactions, agent decisions
- **Layer 3 - Codebase Knowledge**: Generated code, patterns, documentation
- **Layer 4 - Stack Dependencies**: Technology choices, library documentation

**Storage Strategy**:
- SQLite for development, PostgreSQL for production
- Vector embeddings for semantic search
- Hierarchical indexing for efficient retrieval
- Automatic cleanup and archiving policies

### Task 2.3: State Management Service

**File**: `core/services/state_service.py`

**Core Functionality**:
```python
class StateService:
    def __init__(self):
        self.agent_states = {}
        self.workflow_states = {}
        self.conversation_state = None
    
    async def update_agent_state(self, agent_id: str, state: dict):
        """Update individual agent state"""
        pass
    
    async def get_workflow_status(self, workflow_id: str):
        """Get current workflow execution status"""
        pass
    
    async def checkpoint_state(self, checkpoint_id: str):
        """Create state checkpoint for recovery"""
        pass
```

**Features**:
- Real-time agent state tracking
- Workflow progress monitoring
- State persistence and recovery
- Conflict resolution for concurrent operations
- Event-driven state updates

### Task 2.4: Search Service Integration

**File**: `core/services/search_service.py`

**Multi-Provider Search**:
```python
class SearchService:
    def __init__(self):
        self.providers = {
            'tavily': TavilyProvider(),
            'exa': ExaProvider(),
            'fallback': DuckDuckGoProvider()
        }
    
    async def search_knowledge(self, query: str, search_type: str = 'general'):
        """Perform knowledge search with provider fallback"""
        pass
    
    async def deep_research(self, topic: str, max_sources: int = 10):
        """Conduct comprehensive research on topic"""
        pass
```

**Search Types**:
- **General Knowledge**: Broad topic research
- **Technical Documentation**: API docs, tutorials
- **Best Practices**: Industry standards, patterns
- **Code Examples**: Implementation references

## Backend API Enhancements

### Task 2.5: Service Integration Endpoints

**File**: `app/api/app.py`

**New Endpoints**:
```python
# Memory management
@app.post("/api/memory/store")
async def store_memory(request: MemoryStoreRequest):
    pass

@app.get("/api/memory/retrieve")
async def retrieve_memory(query: str, layers: list = None):
    pass

# State management
@app.get("/api/state/agents")
async def get_agent_states():
    pass

@app.get("/api/state/workflow/{workflow_id}")
async def get_workflow_state(workflow_id: str):
    pass

# Search services
@app.post("/api/search/knowledge")
async def search_knowledge(request: SearchRequest):
    pass
```

## Frontend Integration

### Task 2.6: Service Integration Components

**File**: `frontend/lib/api/core-services.ts`

**Service Clients**:
```typescript
export class CoreServicesClient {
  async storeMemory(layer: string, data: any, metadata: any) {
    // Store information in memory layer
  }
  
  async retrieveMemory(query: string, layers?: string[]) {
    // Retrieve relevant information
  }
  
  async getAgentStates() {
    // Get current agent states
  }
  
  async searchKnowledge(query: string, searchType: string) {
    // Perform knowledge search
  }
}
```

**Frontend Components**:
- **Memory Browser**: View and search stored knowledge
- **State Monitor**: Real-time agent and workflow status
- **Search Interface**: Integrated search functionality

## Testing Strategy

### Task 2.7: Service Testing

**Unit Tests**:
```python
# test_memory_service.py
class TestMemoryService:
    async def test_store_and_retrieve(self):
        pass
    
    async def test_layer_isolation(self):
        pass

# test_state_service.py
class TestStateService:
    async def test_agent_state_updates(self):
        pass
    
    async def test_workflow_tracking(self):
        pass
```

**Integration Tests**:
- Service interaction testing
- API endpoint validation
- Frontend-backend communication
- Error handling and recovery

## Configuration Updates

### Task 2.8: Enhanced Configuration

**File**: `core/config.py`

**New Configuration Sections**:
```python
class Settings:
    # LLM Configuration
    GROQ_API_KEY: str
    GEMINI_API_KEY: str
    OPENAI_API_KEY: str
    
    # Search Configuration
    TAVILY_API_KEY: str
    EXA_API_KEY: str
    
    # Memory Configuration
    MEMORY_DB_URL: str
    VECTOR_DB_URL: str
    
    # State Management
    STATE_PERSISTENCE: bool = True
    CHECKPOINT_INTERVAL: int = 300  # seconds
```

## Validation Criteria

### Backend Validation
- [ ] All services initialize without errors
- [ ] Memory layers store and retrieve data correctly
- [ ] State management tracks changes accurately
- [ ] Search services return relevant results
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] Service clients connect successfully
- [ ] Real-time updates display correctly
- [ ] Error states handled gracefully
- [ ] User interactions trigger appropriate backend calls

### Integration Validation
- [ ] Services communicate without conflicts
- [ ] State consistency maintained across operations
- [ ] Memory retrieval supports agent decision-making
- [ ] Search results integrate into workflows

## Human Testing Scenarios

1. **Memory Storage Test**: Store sample knowledge and verify retrieval
2. **State Monitoring Test**: Monitor agent states during simple operations
3. **Search Integration Test**: Perform searches and view results in UI
4. **Service Fallback Test**: Test LLM fallback mechanisms
5. **Error Recovery Test**: Verify graceful handling of service failures

## Next Steps

After successful validation of core services, proceed to **03-agent-framework-enhancement.md** for implementing the enhanced agent framework that will utilize these services.

---

**Dependencies**: This phase builds upon the existing codebase and requires the enhanced services to be functional before proceeding to agent implementation.