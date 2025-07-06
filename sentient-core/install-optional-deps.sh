#!/bin/bash

# Script to install optional dependencies at runtime
# This allows the container to start quickly and install heavy dependencies only when needed

echo "Installing optional dependencies..."

# Install vector database and search dependencies
pip install --no-cache-dir \
    chromadb \
    sentence-transformers \
    faiss-cpu

# Install search APIs
pip install --no-cache-dir \
    tavily-python \
    exa-py \
    duckduckgo-search

# Install additional AI frameworks
pip install --no-cache-dir \
    pyautogen

echo "Optional dependencies installed successfully!" 