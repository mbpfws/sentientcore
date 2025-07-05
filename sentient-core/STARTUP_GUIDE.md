# Sentient Core - Startup Guide

This guide will help you set up and test the Sentient Core system, specifically Build 1 and Build 2 implementations as outlined in the project documentation.

## Quick Start

### First Time Setup
1. Run `start-full-system.bat` - This will:
   - Check Python and Node.js installations
   - Install all backend dependencies (pip install -r requirements.txt)
   - Install all frontend dependencies (npm install)
   - Verify environment files exist
   - Start both backend and frontend servers
   - Open the application in your browser

### Subsequent Runs
1. Run `start-servers.bat` - This will:
   - Quickly start both servers without reinstalling dependencies
   - Open the application in your browser

### Stopping Servers
1. Run `stop-servers.bat` - This will:
   - Cleanly shut down both backend and frontend servers
   - Kill any remaining processes

## System Requirements

- **Python 3.8+** (with pip)
- **Node.js 18+** (with npm)
- **Windows OS** (batch files are Windows-specific)

## Environment Setup

### Backend Environment (.env)
The system requires API keys for various services. The `.env` file should contain:
```
GROQ_API_KEY="your_groq_api_key"
GOOGLE_API_KEY="your_google_api_key"
GEMINI_API_KEY="your_gemini_api_key"
E2B_API_KEY="your_e2b_api_key"
EXA_API_KEY="your_exa_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

### Frontend Environment (frontend/.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Testing Build 1 & Build 2 Implementations

### Build 1: Core Conversation & Orchestration Loop
**Goal**: Test persistent, stateful conversational agent

**Test Steps**:
1. Open http://localhost:3000
2. Send simple messages like:
   - "Hello, how are you?"
   - "What is your purpose?"
   - "What was the first thing I said?"

**Expected Results**:
- Agent responds contextually to each message
- Agent remembers previous messages in the session
- Conversation history is maintained
- Frontend displays back-and-forth conversation

### Build 2: Research Agent & Tool Use
**Goal**: Test delegation to Research Agent with external search

**Test Steps**:
1. Send research-oriented prompts like:
   - "I want a solution for my weak English. Can you research some popular language learning apps?"
   - "Find information about the best Python web frameworks"
   - "What are the latest trends in AI development?"

**Expected Results**:
- Orchestrator identifies research intent
- Research Agent is activated
- External search tools (Tavily/Exa) are used
- Summarized research results are returned
- Conversation can continue with follow-up questions

## System Architecture

### Backend (FastAPI)
- **Port**: 8000
- **API Documentation**: http://localhost:8000/docs
- **Main Components**:
  - Ultra Orchestrator Agent
  - Research Agent
  - Memory Management System
  - LLM Integration (Groq, Google Gemini)
  - External Search Integration

### Frontend (Next.js)
- **Port**: 3000
- **Main Features**:
  - Real-time chat interface
  - Research results display
  - Session management
  - Verbose logging panel
  - Export functionality

## Troubleshooting

### Common Issues

1. **"Python is not installed or not in PATH"**
   - Install Python 3.8+ from python.org
   - Add Python to your system PATH

2. **"Node.js is not installed or not in PATH"**
   - Install Node.js 18+ from nodejs.org
   - Add Node.js to your system PATH

3. **"Failed to install dependencies"**
   - Check your internet connection
   - Try running as administrator
   - Clear npm cache: `npm cache clean --force`

4. **"Backend server won't start"**
   - Check if port 8000 is already in use
   - Verify .env file has correct API keys
   - Check Python dependencies are installed

5. **"Frontend server won't start"**
   - Check if port 3000 is already in use
   - Verify frontend/.env.local exists
   - Check Node.js dependencies are installed

### Manual Commands

If batch files don't work, you can run manually:

**Backend**:
```bash
cd d:\sentientcore\sentient-core
pip install -r requirements.txt
uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend**:
```bash
cd d:\sentientcore\sentient-core\frontend
npm install
npm run dev
```

## Monitoring & Debugging

- **Backend Logs**: Check the backend server window for API logs
- **Frontend Logs**: Check browser developer console
- **API Testing**: Use http://localhost:8000/docs for direct API testing
- **Network Issues**: Verify both servers are running and accessible

## Next Steps

After confirming Build 1 and Build 2 work correctly:
1. Test the monitoring agent verbose output
2. Verify memory persistence across sessions
3. Test research agent with different query types
4. Explore the API documentation for advanced features

For issues or questions, check the project documentation in `/project-doc/` folder.