#!/bin/bash
set -e

# Run the FastAPI web app (uvicorn dev server with live reload) from the bind-mounted source.
exec python3 -m uvicorn web_app:app --reload --host 0.0.0.0 --port 3000
