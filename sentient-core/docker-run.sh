#!/bin/bash

# Helper script for Docker commands

# Function to display help message
show_help() {
    echo "Sentient Core Docker Helper Script"
    echo ""
    echo "Usage: ./docker-run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start       - Start all services in production mode"
    echo "  start:dev   - Start all services in development mode"
    echo "  start:alt   - Start using alternative pip-based Dockerfile"
    echo "  start:minimal - Start with minimal dependencies (fastest build)"
    echo "  stop        - Stop all running services"
    echo "  restart     - Restart all services"
    echo "  build       - Rebuild all services"
    echo "  build:dev   - Rebuild all services in development mode"
    echo "  build:alt   - Rebuild using alternative pip-based Dockerfile"
    echo "  build:minimal - Rebuild with minimal dependencies"
    echo "  build:timeout - Rebuild with increased timeout settings"
    echo "  logs        - View logs from all services"
    echo "  logs:back   - View logs from backend service"
    echo "  logs:front  - View logs from frontend service"
    echo "  logs:ui     - View logs from Streamlit UI service"
    echo "  clean       - Remove all containers and volumes"
    echo "  help        - Show this help message"
    echo ""
    echo "Build Time Optimization:"
    echo "  For fastest builds, use 'start:minimal' or 'build:minimal'"
    echo "  This installs only essential dependencies and skips heavy ML packages"
    echo ""
    echo "Timeout Issues:"
    echo "  If you encounter network timeout issues during build, try:"
    echo "  1. Use 'build:timeout' command"
    echo "  2. Use 'start:alt' to use the pip-based Dockerfile"
    echo "  3. Use 'start:minimal' for fastest build"
    echo "  4. Make sure you have a stable internet connection"
    echo ""
}

# Check if command is provided
if [ -z "$1" ]; then
    show_help
    exit 1
fi

# Handle commands
case "$1" in
    start)
        echo "Starting services in production mode..."
        docker-compose up -d
        ;;
    start:dev)
        echo "Starting services in development mode..."
        docker-compose -f docker-compose.dev.yml up -d
        ;;
    start:alt)
        echo "Starting services using alternative pip-based Dockerfile..."
        docker-compose -f docker-compose.yml -f docker-compose.alt.yml up -d
        ;;
    start:minimal)
        echo "Starting services with minimal dependencies (fastest build)..."
        docker-compose -f docker-compose.minimal.yml up -d
        ;;
    stop)
        echo "Stopping all services..."
        docker-compose down
        ;;
    restart)
        echo "Restarting all services..."
        docker-compose restart
        ;;
    build)
        echo "Rebuilding all services..."
        docker-compose build
        ;;
    build:dev)
        echo "Rebuilding all services in development mode..."
        docker-compose -f docker-compose.dev.yml build
        ;;
    build:alt)
        echo "Rebuilding using alternative pip-based Dockerfile..."
        docker-compose -f docker-compose.yml -f docker-compose.alt.yml build
        ;;
    build:minimal)
        echo "Rebuilding with minimal dependencies..."
        docker-compose -f docker-compose.minimal.yml build
        ;;
    build:timeout)
        echo "Rebuilding with increased timeout settings..."
        DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose build \
            --build-arg PIP_DEFAULT_TIMEOUT=100 \
            --build-arg BUILDKIT_STEP_LOG_MAX_SIZE=10485760
        ;;
    logs)
        echo "Viewing logs from all services..."
        docker-compose logs -f
        ;;
    logs:back)
        echo "Viewing logs from backend service..."
        docker-compose logs -f backend
        ;;
    logs:front)
        echo "Viewing logs from frontend service..."
        docker-compose logs -f frontend
        ;;
    logs:ui)
        echo "Viewing logs from Streamlit UI service..."
        docker-compose logs -f streamlit
        ;;
    clean)
        echo "Removing all containers and volumes..."
        docker-compose down -v
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 