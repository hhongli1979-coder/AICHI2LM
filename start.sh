#!/bin/bash
# TeleChat Container Entrypoint Script
# This script detects available entry points and starts the appropriate service.
# Priority order: uvicorn api.proxy:app > uvicorn main:app > deploy.py > health server

set -e

echo "=========================================="
echo "TeleChat Container Starting..."
echo "=========================================="

# Function to check if a Python module/file exists
check_module() {
    python -c "import $1" 2>/dev/null
    return $?
}

check_file() {
    [ -f "$1" ]
    return $?
}

# Try to start uvicorn with api.proxy:app (highest priority)
if check_file "api/proxy.py" || check_file "api/__init__.py"; then
    echo "✓ Found api.proxy module, starting uvicorn api.proxy:app..."
    exec uvicorn api.proxy:app --host 0.0.0.0 --port 8000
fi

# Try to start uvicorn with main:app
if check_file "main.py"; then
    echo "✓ Found main.py, checking for FastAPI app..."
    if grep -q "FastAPI" main.py 2>/dev/null; then
        echo "✓ Starting uvicorn main:app..."
        exec uvicorn main:app --host 0.0.0.0 --port 8000
    fi
fi

# Try to run deploy.py
if check_file "deploy.py"; then
    echo "✓ Found deploy.py, starting deployment script..."
    exec python deploy.py
fi

# Try to start the service from service/telechat_service.py
if check_file "service/telechat_service.py"; then
    echo "✓ Found service/telechat_service.py, starting TeleChat service..."
    cd service
    exec uvicorn telechat_service:app --host 0.0.0.0 --port 8000
fi

# Fallback: Start a simple health check server
echo "⚠ No recognized entry point found, starting fallback health server..."
echo "Available endpoints:"
echo "  GET /health - Health check"
echo "  GET /      - Service info"

python -c "
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'healthy', 'message': 'TeleChat container is running but no main service was detected'}

@app.get('/')
def root():
    return {
        'service': 'TeleChat',
        'status': 'running',
        'message': 'Container started successfully, but no main application entry point was found.',
        'instructions': 'Please check the deployment configuration or mount a valid TeleChat application.'
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
"
