#!/bin/bash
# start.sh - Docker entrypoint detection script
# This script tries different startup methods in order of preference

set -e

echo "Starting TeleChat service..."

# Check if API proxy module exists
if python -c "import api.proxy" 2>/dev/null; then
    echo "Starting with uvicorn api.proxy:app..."
    exec uvicorn api.proxy:app --host 0.0.0.0 --port 8000
fi

# Check if main module exists
if python -c "import main" 2>/dev/null; then
    echo "Starting with uvicorn main:app..."
    exec uvicorn main:app --host 0.0.0.0 --port 8000
fi

# Check if deploy.py exists
if [ -f "deploy.py" ]; then
    echo "Starting with python deploy.py..."
    exec python deploy.py
fi

# Fallback: simple health check server
echo "No known entrypoint found. Starting fallback health server..."
python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({'status': 'ok', 'message': 'TeleChat container is running'})
            self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        print(f'{self.address_string()} - {format % args}')

print('Fallback health server listening on port 8000...')
HTTPServer(('0.0.0.0', 8000), HealthHandler).serve_forever()
"
