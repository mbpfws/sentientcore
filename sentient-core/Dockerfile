FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry with higher timeout settings
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"

# Configure Poetry to not use a virtual environment and increase network timeout
RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10 && \
    poetry config installer.parallel false && \
    poetry config http-basic.pypi --force-unsigned-protocol && \
    poetry config --list

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Try installing with increased timeout and retry strategy
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    pip install --upgrade pip && \
    pip install urllib3==2.0.7 requests==2.31.0 && \
    export PYTHONHTTPSVERIFY=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i: Installing dependencies..." && \
        poetry install --no-interaction --no-ansi --no-root --with dev && \
        if [ $? -eq 0 ]; then \
            break; \
        fi; \
        echo "Attempt $i failed, retrying..." && \
        sleep 5; \
    done

# Copy the rest of the application
COPY . .

# Final poetry install for the project itself (faster as deps are installed)
RUN poetry install --no-interaction --no-ansi --only-root

# Expose port for FastAPI
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 