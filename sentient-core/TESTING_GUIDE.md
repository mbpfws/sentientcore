# SentientCore Full-Stack Testing Guide

## 🚀 System Status

### Backend (FastAPI)
- **Status**: ✅ Running
- **URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Frontend (Next.js)
- **Status**: ✅ Running  
- **URL**: http://localhost:3000
- **Framework**: Next.js 14.1.0 with App Router

## 🧪 Testing Checklist

### 1. Backend API Testing

#### Core Endpoints
- [ ] **Health Check**: `GET /health`
- [ ] **API Documentation**: Visit `/docs` for Swagger UI
- [ ] **Memory Management**: `POST /api/memory/store` and `GET /api/memory/retrieve`
- [ ] **State Management**: `GET /api/state` and `POST /api/state`
- [ ] **Agent Execution**: `POST /api/agents/execute`

#### Test Commands
```bash
# Health check
curl http://localhost:8000/health

# Test memory storage
curl -X POST http://localhost:8000/api/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "Test memory", "metadata": {"type": "test"}}'

# Test state retrieval
curl http://localhost:8000/api/state
```

### 2. Frontend Testing

#### UI Components
- [ ] **Landing Page**: Check if main page loads
- [ ] **Navigation**: Test menu and routing
- [ ] **Agent Interface**: Test agent interaction components
- [ ] **Memory Management UI**: Test memory storage/retrieval interface
- [ ] **State Visualization**: Check state management dashboard

#### Integration Testing
- [ ] **API Connectivity**: Frontend → Backend communication
- [ ] **Real-time Updates**: WebSocket connections (if implemented)
- [ ] **Error Handling**: Test error states and user feedback

### 3. Agent System Testing

#### Available Agents
- [ ] **Research Agent**: Test research capabilities
- [ ] **Architect Planner**: Test planning functionality
- [ ] **Frontend Developer**: Test frontend code generation
- [ ] **Backend Developer**: Test backend code generation
- [ ] **Monitoring Agent**: Test system monitoring

#### Workflow Testing
- [ ] **Single Agent Execution**: Test individual agent responses
- [ ] **Multi-Agent Workflows**: Test agent collaboration
- [ ] **State Persistence**: Verify state is maintained across requests
- [ ] **Memory Integration**: Test agent memory storage/retrieval

### 4. Database & Storage Testing

#### SQLite Databases
- [ ] **Memory Management DB**: Check `memory_management.db`
- [ ] **State Management DB**: Check `state_management.db`
- [ ] **Vector Storage**: Test vector database operations

#### Test Queries
```python
# Test database connections
import sqlite3

# Memory DB
conn = sqlite3.connect('memory_management.db')
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Memory tables:", cursor.fetchall())

# State DB  
conn = sqlite3.connect('state_management.db')
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("State tables:", cursor.fetchall())
```

## 🐛 Common Issues & Troubleshooting

### Backend Issues
1. **Import Errors**: Check if all dependencies are installed
2. **Database Locks**: Restart if SQLite databases are locked
3. **Port Conflicts**: Ensure port 8000 is available

### Frontend Issues
1. **Build Errors**: Check Node.js version compatibility
2. **API Connection**: Verify backend is running on port 8000
3. **CORS Issues**: Check CORS configuration in FastAPI

### Agent Issues
1. **LLM API Keys**: Verify Groq and Gemini API keys in `.env`
2. **Memory Persistence**: Check database write permissions
3. **State Synchronization**: Verify state service is running

## 📊 Performance Testing

### Load Testing
```bash
# Install Apache Bench
apt-get install apache2-utils

# Test API performance
ab -n 100 -c 10 http://localhost:8000/health
```

### Memory Usage
```python
# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

## 🔧 Development Tools

### Backend Development
- **FastAPI Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Database Browser**: Use SQLite browser for database inspection

### Frontend Development
- **Next.js Dev Tools**: Built-in development features
- **React DevTools**: Browser extension for React debugging
- **Network Tab**: Monitor API calls in browser DevTools

## 📝 Bug Reporting Template

When reporting issues, please include:

```markdown
### Bug Report

**Component**: [Backend/Frontend/Agent/Database]
**Severity**: [Critical/High/Medium/Low]
**Environment**: [Development/Testing]

**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Behavior**:

**Actual Behavior**:

**Error Messages**:
```

**Console Logs**:
```

```

**Additional Context**:

```

## 🎯 Next Steps

1. **Start Testing**: Begin with health checks and basic API endpoints
2. **Report Issues**: Use the bug report template above
3. **Iterative Fixes**: I'll address issues as you discover them
4. **Feature Testing**: Test specific features you're most interested in
5. **Performance Optimization**: Once basic functionality is verified

---

**Happy Testing! 🚀**

Both servers are running and ready for comprehensive testing. Start with the basic health checks and work your way through the checklist above.