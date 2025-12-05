#!/usr/bin/env bash
set -euo pipefail

# start.sh: entrypoint detection
# Priority: api.proxy -> main:app -> deploy.py -> fallback health

echo "Container start: detecting entrypoint..."

# 1) api.proxy
if python -c "import importlib,sys; sys.path.insert(0,'/app'); print(importlib.util.find_spec('api.proxy') is not None)" 2>/dev/null | grep -q True; then
  echo "Starting uvicorn api.proxy:app"
  exec uvicorn api.proxy:app --host 0.0.0.0 --port 8000 --workers 1
fi

# 2) main:app
if python -c "import importlib,sys; sys.path.insert(0,'/app'); print(importlib.util.find_spec('main') is not None)" 2>/dev/null | grep -q True; then
  echo "Starting uvicorn main:app"
  exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
fi

# 3) deploy.py
if [ -f /app/deploy.py ]; then
  echo "Found deploy.py; running python deploy.py"
  exec python /app/deploy.py
fi

# 4) fallback
echo "No web entrypoint found. Starting fallback health server on :8000"
python - <<'PY'
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.end_headers()
        self.wfile.write(b'{"status":"no-entrypoint","msg":"No recognized web entrypoint found. Edit start.sh if needed."}')
HTTPServer(('0.0.0.0',8000), H).serve_forever()
PY
