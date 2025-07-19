#!/bin/bash

# ARC MCP Server Wrapper Script
# Ensures proper environment and starts the custom ARC MCP server

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Change to project root directory
cd "$PROJECT_ROOT"

# Ensure virtual environment is activated
if [[ "$VIRTUAL_ENV" != "$PROJECT_ROOT/venv" ]]; then
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    else
        echo "Error: Virtual environment not found at $PROJECT_ROOT/venv"
        echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
fi

# Ensure environment variables are loaded
if [[ -f ".env" ]]; then
    # Export .env variables without overriding existing ones
    set -a
    source .env
    set +a
fi

# Verify Neo4j is running
if ! nc -z localhost 7687 2>/dev/null; then
    echo "Warning: Neo4j server not accessible on localhost:7687"
    echo "Make sure Neo4j is running: brew services start neo4j"
fi

# Change to tools directory and run the MCP server
cd tools
exec python arc_mcp_server.py "$@" 