# Sentient Core Docker Setup Guide

This guide provides comprehensive instructions for setting up and running the Sentient Core application using Docker, including solutions for common issues and optimization strategies.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Setup Options](#docker-setup-options)
3. [Environment Configuration](#environment-configuration)
4. [Build Optimization](#build-optimization)
5. [Troubleshooting](#troubleshooting)
6. [Development Workflow](#development-workflow)
7. [Production Deployment](#production-deployment)

## Quick Start

### Prerequisites

- Docker Desktop installed and running
- At least 4GB of available RAM
- Stable internet connection

### Fastest Setup (Recommended)

For the quickest start with minimal build time:

```bash
# Navigate to the project directory
cd /Users/mac/sentientcore/sentient-core

# Start with minimal dependencies (builds in ~5 minutes)
./docker-run.sh start:minimal
```

This will:
- Build the backend with only essential dependencies
- Start the frontend service
- Skip heavy ML packages that cause long build times
- Make the application available at:
  - Frontend: http://localhost:3000
  - Backend API: http://localhost:8000

## Docker Setup Options

### 1. Minimal Setup (Fastest - Recommended)

**Use Case**: Quick development, testing, or when you don't need advanced ML features.

**Build Time**: ~5 minutes

```bash
./docker-run.sh start:minimal
```

**Includes**:
- FastAPI web server
- LangChain and LangGraph
- Groq and Google Generative AI
- Basic utilities (numpy, requests, httpx)

**Excludes**:
- sentence-transformers (large ML model)
- chromadb (vector database)
- faiss-cpu (vector search)
- tavily-python, exa-py (search APIs)

### 2. Full Setup (Complete Features)

**Use Case**: Production deployment with all features.

**Build Time**: 30-60 minutes (depending on network)

```bash
./docker-run.sh start
```

**Includes**: All dependencies and features.

### 3. Development Setup (Hot Reloading)

**Use Case**: Active development with code changes.

```bash
./docker-run.sh start:dev
```

**Features**:
- Hot reloading for both frontend and backend
- Volume mounting for live code changes
- Debug mode enabled

### 4. Alternative Setup (Pip-based)

**Use Case**: When Poetry causes issues.

```bash
./docker-run.sh start:alt
```

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# LLM API Keys (Required)
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Search API Keys (Optional - for advanced search features)
TAVILY_API_KEY=your_tavily_api_key_here
EXA_API_KEY=your_exa_api_key_here

# E2B API Key (Optional - for code execution)
E2B_API_KEY=your_e2b_api_key_here

# Database Configuration (Optional - defaults to SQLite)
DATABASE_URL=sqlite:///./memory_management.db
VECTOR_DB_PATH=./memory_vectors

# Development Settings (Optional)
DEBUG=1
PYTHONUNBUFFERED=1
```

### Getting API Keys

1. **Groq API Key**: Sign up at https://console.groq.com/
2. **Google API Key**: Get from Google Cloud Console
3. **Tavily API Key**: Sign up at https://tavily.com/
4. **Exa API Key**: Sign up at https://exa.ai/

## Build Optimization

### Why Build Times Were Long

The original build was taking 50+ minutes due to:

1. **Heavy ML Packages**:
   - `sentence-transformers` (~1.5GB)
   - `chromadb` (~500MB)
   - `faiss-cpu` (~200MB)

2. **Network Issues**:
   - Timeouts during package downloads
   - Unstable connections to PyPI

3. **Poetry Dependencies**:
   - Complex dependency resolution
   - Network timeouts during installation

### Solutions Implemented

#### 1. Minimal Dependencies Approach

Created `requirements-minimal.txt` with only essential packages:

```txt
fastapi
uvicorn[standard]
pydantic
python-dotenv
langchain
langgraph
groq
google-generativeai
numpy
requests
httpx
```

#### 2. Multi-Stage Builds

Implemented `Dockerfile.optimized` with:
- Separate dependency installation stage
- Better layer caching
- Reduced final image size

#### 3. Runtime Dependency Installation

Created `install-optional-deps.sh` for installing heavy packages when needed:

```bash
# Inside running container
docker exec -it sentient-core-backend-minimal bash
./install-optional-deps.sh
```

#### 4. Build Configuration

Added Docker BuildKit optimizations:
- Increased timeout settings
- Better caching strategies
- Parallel build support

## Troubleshooting

### Common Issues and Solutions

#### 1. Build Timeout Issues

**Symptoms**: Build fails with "Read timed out" or "Connection error"

**Solutions**:
```bash
# Try minimal setup first
./docker-run.sh start:minimal

# Use timeout-optimized build
./docker-run.sh build:timeout

# Use alternative pip-based approach
./docker-run.sh start:alt
```

#### 2. Frontend Build Errors

**Symptoms**: TypeScript compilation errors

**Solutions**:
```bash
# Check for missing types (already fixed in this setup)
# The missing AgentState, WorkflowState, etc. types have been added

# Rebuild frontend only
docker-compose build frontend

# Check logs for specific errors
./docker-run.sh logs:front
```

#### 3. Memory Issues

**Symptoms**: Container crashes or slow performance

**Solutions**:
```bash
# Increase Docker memory limit in Docker Desktop
# Recommended: 4GB minimum, 8GB preferred

# Check container resource usage
docker stats

# Restart with fresh containers
./docker-run.sh clean
./docker-run.sh start:minimal
```

#### 4. Network Issues

**Symptoms**: Cannot connect to services or API calls fail

**Solutions**:
```bash
# Check if services are running
./docker-run.sh logs

# Verify ports are not in use
lsof -i :3000
lsof -i :8000

# Restart services
./docker-run.sh restart
```

#### 5. Permission Issues

**Symptoms**: "Permission denied" errors

**Solutions**:
```bash
# Fix script permissions
chmod +x docker-run.sh
chmod +x install-optional-deps.sh

# Run with sudo if needed (not recommended for production)
sudo ./docker-run.sh start:minimal
```

### Debugging Commands

```bash
# View all container logs
./docker-run.sh logs

# View specific service logs
./docker-run.sh logs:back
./docker-run.sh logs:front

# Check container status
docker-compose ps

# Access container shell
docker exec -it sentient-core-backend-minimal bash

# Check Docker system info
docker system df
docker system prune
```

## Development Workflow

### Local Development

1. **Start Development Environment**:
   ```bash
   ./docker-run.sh start:dev
   ```

2. **Make Code Changes**: Files are mounted for hot reloading

3. **View Logs**: Monitor changes in real-time
   ```bash
   ./docker-run.sh logs:back
   ```

4. **Test Changes**: Access at http://localhost:3000

### Adding Features

1. **For Basic Features**: Use minimal setup
2. **For ML Features**: Install optional dependencies
   ```bash
   docker exec -it sentient-core-backend-minimal bash
   ./install-optional-deps.sh
   ```

3. **For New Dependencies**: Update requirements files
   - `requirements-minimal.txt` for essential packages
   - `requirements.txt` for full feature set

### Testing

```bash
# Run backend tests
docker exec -it sentient-core-backend-minimal poetry run pytest

# Run frontend tests
docker exec -it sentient-core-frontend npm test
```

## Production Deployment

### Recommended Production Setup

1. **Use Full Setup**: For complete feature set
   ```bash
   ./docker-run.sh start
   ```

2. **Environment Variables**: Set all required API keys
3. **Resource Allocation**: Ensure adequate CPU and memory
4. **Monitoring**: Set up logging and health checks

### Production Considerations

1. **Security**:
   - Use secrets management for API keys
   - Enable HTTPS
   - Configure firewall rules

2. **Performance**:
   - Use production-grade database
   - Configure caching
   - Set up load balancing

3. **Monitoring**:
   - Health check endpoints
   - Log aggregation
   - Metrics collection

### Deployment Commands

```bash
# Production build
./docker-run.sh build

# Start production services
./docker-run.sh start

# Monitor production
./docker-run.sh logs

# Update production
git pull
./docker-run.sh build
./docker-run.sh restart
```

## File Structure

```
sentient-core/
├── Dockerfile                    # Main Dockerfile (Poetry-based)
├── Dockerfile.alt               # Alternative (pip-based)
├── Dockerfile.minimal           # Minimal dependencies
├── Dockerfile.optimized         # Multi-stage build
├── docker-compose.yml           # Production setup
├── docker-compose.dev.yml       # Development setup
├── docker-compose.minimal.yml   # Minimal setup
├── docker-compose.alt.yml       # Alternative setup
├── requirements.txt             # Full dependencies
├── requirements-minimal.txt     # Minimal dependencies
├── docker-run.sh               # Helper script
├── install-optional-deps.sh    # Runtime dependency installer
├── DOCKER.md                   # Basic documentation
└── DOCKER_SETUP_GUIDE.md       # This comprehensive guide
```

## Support

If you encounter issues not covered in this guide:

1. Check the logs: `./docker-run.sh logs`
2. Verify your environment setup
3. Ensure Docker has adequate resources
4. Try the minimal setup first
5. Check the troubleshooting section above

For additional help, refer to the project's main documentation or create an issue in the repository. 