# Sentient Core Migration Process

## Introduction
This document outlines the process of migrating the Sentient Core multi-agent RAG system from Streamlit to a modular full-stack architecture using Next.js 15 and FastAPI.

## Architecture Overview

### Backend (FastAPI)
- **Directory**: `/app`
- **Main Application**: `api/app.py`
- **API Routers**:
  - `api/routers/agents.py` - Endpoints for agent management and capabilities
  - `api/routers/workflows.py` - Endpoints for workflow execution and task management
  - `api/routers/chat.py` - Endpoints for chat messaging and history

### Frontend (Next.js 15)
- **Directory**: `/frontend`
- **Main Components**:
  - App layout and routing (`app/layout.tsx`, `app/page.tsx`)
  - UI components (`components/`)
  - API services (`lib/api/`)
  - Global state management (`lib/context/app-context.tsx`)

## Migration Strategy

### Incremental Feature Migration
The migration follows an incremental approach, where each feature is migrated one by one:

1. Extract backend logic for a feature into FastAPI endpoints
2. Implement corresponding frontend components using Next.js and Shadcn UI
3. Integrate backend and frontend via API services
4. Write unit tests for both backend and frontend
5. Validate the feature end-to-end

### Features Migrated So Far

#### Chat Interface
- **Backend**: FastAPI endpoints for message processing and chat history
- **Frontend**: Chat interface with message input, research mode options, and message display
- **Integration**: Real-time messaging between frontend and backend

## Development Environment Setup

### Prerequisites
- Node.js 18+
- Python 3.10+
- npm or yarn

### Installation

#### Backend Setup
```bash
cd app
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd frontend
# Run the setup script to install all dependencies
./setup.bat
```

### Running the Application

#### Option 1: Individually
Backend:
```bash
cd app
uvicorn api.app:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm run dev
```

#### Option 2: Using start-dev script
Run both frontend and backend with a single command:
```bash
./start-dev.bat
```

### Access Points
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Testing

### Backend Tests
```bash
cd app
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Environment Variables

### Frontend (.env.local)
- `NEXT_PUBLIC_API_URL`: URL for the FastAPI backend (default: http://localhost:8000)

### Backend
- Environment variables will be added as needed during further migration

## Next Steps
- Continue migrating remaining features from Streamlit application
- Enhance error handling and logging
- Implement authentication and authorization
- Optimize performance and responsiveness
