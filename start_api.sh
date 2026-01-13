#!/bin/bash

# IAAIR Paper Ingestion API Startup Script

echo "Starting IAAIR Paper Ingestion API..."

# Get the current directory and find the correct path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# If we're already in pipelines/ingestions, go to root
if [[ "$PWD" == */pipelines/ingestions ]]; then
    cd ../..
elif [[ "$PWD" == */IAAIR ]]; then
    # We're in the root directory
    cd .
else
    # Navigate to script directory and then to root
    cd "$SCRIPT_DIR"
fi

echo "Working from directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

# Navigate to the ingestions directory
cd pipelines/ingestions

# Start the FastAPI server
echo "Starting FastAPI server on http://localhost:8000"
echo "API Documentation available at: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload