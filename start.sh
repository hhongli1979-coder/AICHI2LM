#!/bin/bash
# TeleChat Docker Entrypoint Script
# Smart service detection and startup

set -e

echo "🚀 TeleChat Docker Container Starting..."
echo "📍 Working directory: $(pwd)"
echo "🔍 Detecting available services..."

# Function to check if a Python module/file exists
check_python_module() {
    python -c "import $1" 2>/dev/null
}

check_python_file() {
    [ -f "$1" ]
}

# Try different startup methods in order of preference

# 1. Try uvicorn with api.proxy:app
if check_python_file "api/proxy.py" || check_python_file "api/__init__.py"; then
    echo "✅ Found api.proxy module"
    echo "🌟 Starting with: uvicorn api.proxy:app --host 0.0.0.0 --port 8000"
    exec uvicorn api.proxy:app --host 0.0.0.0 --port 8000

# 2. Try uvicorn with main:app
elif check_python_file "main.py"; then
    echo "✅ Found main.py"
    echo "🌟 Starting with: uvicorn main:app --host 0.0.0.0 --port 8000"
    exec uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Try running deploy.py
elif check_python_file "deploy.py"; then
    echo "✅ Found deploy.py"
    echo "🌟 Starting with: python deploy.py"
    exec python deploy.py

# 4. Try service/telechat_service.py if it exists
elif check_python_file "service/telechat_service.py"; then
    echo "✅ Found service/telechat_service.py"
    echo "🌟 Starting with: uvicorn service.telechat_service:app --host 0.0.0.0 --port 8000"
    cd service
    exec uvicorn telechat_service:app --host 0.0.0.0 --port 8000

# 5. Fallback: Start a simple health check server
else
    echo "⚠️  No known service entry point found"
    echo "🏥 Starting fallback health check server on port 8000"
    echo "💡 Mount your models and configure MODEL_PATH environment variable"
    
    # Create a simple health check server
    python -c '
import http.server
import socketserver
import json
from datetime import datetime

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {
                "status": "healthy",
                "message": "TeleChat container is running",
                "timestamp": datetime.now().isoformat(),
                "note": "No service entry point configured. Please provide models and configuration."
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[{datetime.now().isoformat()}] {format % args}")

PORT = 8000
with socketserver.TCPServer(("0.0.0.0", PORT), HealthHandler) as httpd:
    print(f"Health check server running on port {PORT}")
    print("Visit http://localhost:8000/health")
    httpd.serve_forever()
'
fi
