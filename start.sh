#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
exec proxy-vertex-openai --host 0.0.0.0 --port 8082 --log-level info
