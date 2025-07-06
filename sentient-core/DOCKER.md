# Docker Setup for Sentient Core

This document provides instructions for running the Sentient Core application using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Environment Variables

Create a `.env` file in the root directory with the following variables (adjust as needed):

```
# LLM API Keys
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

# Search API Keys
TAVILY_API_KEY=your_tavily_api_key
EXA_API_KEY=your_exa_api_key

# E2B API Key
E2B_API_KEY=your_e2b_api_key
```

## Quick Start (Recommended)

For the fastest build time, use the minimal setup:

```bash
./docker-run.sh start:minimal
```

This installs only essential dependencies and should build in under 5 minutes.

## Running in Production Mode

To start all services in production mode:

```bash
docker-compose up -d
```

This will start:
- Backend API on port 8000
- Frontend on port 3000
- Streamlit UI on port 8501

To stop all services:

```bash
docker-compose down
```

## Running in Development Mode

For development with hot reloading:

```bash
docker-compose -f docker-compose.dev.yml up -d
```

## Alternative Dockerfile

If you encounter timeout issues with Poetry during the build process, you can use the alternative Dockerfile that uses pip instead:

```bash
# For production
docker-compose -f docker-compose.yml -f docker-compose.alt.yml up -d

# For development
docker-compose -f docker-compose.dev.yml -f docker-compose.alt.yml up -d
```

## Minimal Setup (Fastest Build)

For the fastest possible build time, use the minimal setup that only installs essential dependencies:

```bash
# Start with minimal dependencies
./docker-run.sh start:minimal

# Or build only
./docker-run.sh build:minimal
```

The minimal setup includes:
- FastAPI and Uvicorn for the web server
- Pydantic for data validation
- LangChain and LangGraph for AI frameworks
- Groq and Google Generative AI for LLM providers
- Basic utilities (numpy, requests, httpx)

**Note**: The minimal setup skips heavy ML packages like `sentence-transformers`, `chromadb`, and `faiss-cpu`. If you need these features, you can install them at runtime using:

```bash
# Inside the running container
docker exec -it sentient-core-backend-minimal bash
./install-optional-deps.sh
```

## Handling Network Timeout Issues

If you experience network timeouts during the build process, you can try the following:

1. Use the minimal setup (fastest):
   ```bash
   ./docker-run.sh start:minimal
   ```

2. Use BuildKit with increased timeout:
   ```bash
   DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose build --build-arg PIP_DEFAULT_TIMEOUT=100
   ```

3. Or use the helper script with increased timeout:
   ```bash
   ./docker-run.sh build:timeout
   ```

4. If behind a corporate proxy, configure Docker to use your proxy:
   Edit or create `/etc/docker/daemon.json` with:
   ```json
   {
     "proxies": {
       "http-proxy": "http://proxy.example.com:3128",
       "https-proxy": "http://proxy.example.com:3128",
       "no-proxy": "localhost,127.0.0.1"
     }
   }
   ```

## Accessing the Services

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- Streamlit UI: http://localhost:8501

## Viewing Logs

```bash
# View logs for all services
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f streamlit
```

## Rebuilding Services

If you make changes to dependencies or Dockerfiles:

```bash
docker-compose build
# or for development
docker-compose -f docker-compose.dev.yml build
# or for minimal setup
./docker-run.sh build:minimal
```

## Data Persistence

The following data is persisted through Docker volumes:
- SQLite database files
- Vector database files
- Environment variables

## Troubleshooting

If you encounter issues:

1. Check if all services are running:
   ```bash
   docker-compose ps
   ```

2. Restart a specific service:
   ```bash
   docker-compose restart backend
   ```

3. Check service logs for errors:
   ```bash
   docker-compose logs -f backend
   ```

4. Rebuild and restart all services:
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

5. Network issues during build:
   - Try using the minimal setup: `./docker-run.sh start:minimal`
   - Try using the alternative Dockerfile (`Dockerfile.alt`)
   - Increase network timeouts as mentioned above
   - Try building with a different network connection
   - Consider using Docker's build cache by running:
     ```bash
     docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1
     ```

6. If you need additional dependencies after using minimal setup:
   ```bash
   # Connect to the running container
   docker exec -it sentient-core-backend-minimal bash
   
   # Install optional dependencies
   ./install-optional-deps.sh
   ``` 