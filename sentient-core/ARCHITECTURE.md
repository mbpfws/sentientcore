# Sentient Core Architecture

## System Architecture Diagram

```mermaid
graph TD
    subgraph "Frontend (Next.js 15)"
        A["App Layout (layout.tsx)"] --> B["Main Page (page.tsx)"]
        B --> C1["Chat Interface Component"]
        B --> C2["Task View Component"]
        B --> C3["Agents List Component"]
        D["Global State (AppContext)"] -.-> B
        D -.-> C1
        D -.-> C2
        D -.-> C3
        E["API Services"]
        E1["AgentService"] --> E
        E2["WorkflowService"] --> E
        E3["ChatService"] --> E
        C1 -.-> E3
        C2 -.-> E2
        C3 -.-> E1
    end

    subgraph "Backend (FastAPI)"
        F["Main App (app.py)"]
        G1["Agents Router"] --> F
        G2["Workflows Router"] --> F
        G3["Chat Router"] --> F
        
        H1["Agent Logic"] --> G1
        H2["Workflow Logic"] --> G2
        H3["Chat Processing Logic"] --> G3
        
        I1["Intelligent RAG Graph"] --> H2
        I2["Multi-Agent RAG Graph"] --> H2
        I3["Orchestration Graph"] --> H2
    end

    E1 <--> G1
    E2 <--> G2
    E3 <--> G3
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant RAG_System

    User->>Frontend: Send message
    Frontend->>Backend: POST /chat/message
    Backend->>RAG_System: Process message with selected workflow
    RAG_System->>Backend: Response generated
    Backend->>Frontend: JSON response
    Frontend->>User: Display message

    User->>Frontend: Select workflow mode
    Frontend->>Backend: GET /chat/history/{workflow_mode}
    Backend->>Frontend: Return chat history
    Frontend->>User: Display chat history
```

## Component Structure

### Frontend

```
frontend/
├── app/
│   ├── layout.tsx           # Main app layout with ThemeProvider and AppProvider
│   ├── page.tsx             # Main page with tabs for chat, tasks, and agents
│   └── globals.css          # Global styles with Tailwind utilities
├── components/
│   ├── ui/                  # Shadcn UI components
│   ├── theme-toggle.tsx     # Theme toggle component
│   ├── chat-interface.tsx   # Chat interface component
│   ├── task-view.tsx        # Task view component
│   └── agents-list.tsx      # Agents list component
├── lib/
│   ├── api/                 # API services
│   │   ├── agent-service.ts  
│   │   ├── workflow-service.ts
│   │   ├── chat-service.ts
│   │   └── index.ts         # API service exports
│   ├── context/
│   │   └── app-context.tsx  # Global app context
│   └── utils/
│       └── cn.ts            # Class name utility function
└── tests/
    └── chat-interface.test.js # Frontend tests
```

### Backend

```
app/
├── api/
│   ├── app.py              # Main FastAPI application
│   ├── routers/
│   │   ├── agents.py       # Endpoints for agents
│   │   ├── workflows.py    # Endpoints for workflows
│   │   └── chat.py         # Endpoints for chat
└── tests/
    └── test_chat_api.py    # Backend tests
```

## Communication Flow

1. **User Interaction**: User interacts with the Next.js frontend (sending messages, changing workflows)
2. **Frontend Processing**: React components update UI and call appropriate API services
3. **API Request**: API service makes HTTP request to FastAPI backend
4. **Backend Processing**: FastAPI routes request to appropriate handler
5. **RAG System Integration**: Backend integrates with RAG system components (graphs, agents)
6. **Response Flow**: Response flows back through the backend to frontend to user

## Next Steps in Migration

1. Migrate remaining features one by one, following the same pattern
2. Add authentication and authorization
3. Improve error handling and logging
4. Optimize performance with caching and lazy loading
5. Add comprehensive testing coverage
