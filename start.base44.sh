#!/bin/bash
set -e

# Run the web view (Flask dev server with live reload) from the bind-mounted source.
exec python3 web_app.py
