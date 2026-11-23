#!/bin/bash

# Get the directory where this script is located
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "$APP_DIR/venv/bin/activate"

# Launch the application
cd "$APP_DIR"
python3 "$APP_DIR/main.py"
