#!/usr/bin/env bash
set -Eeuo pipefail

# YUC Player installer for Ubuntu, Lubuntu and other Debian-based systems.
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: this installer currently supports Debian/Ubuntu-based systems (apt-get)."
    exit 1
fi

echo "==> Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

echo "==> Creating Python virtual environment..."
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi

echo "==> Installing Python packages..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$APP_DIR/requirements.txt"

if [[ -f "$APP_DIR/run.sh" ]]; then
    chmod +x "$APP_DIR/run.sh"
fi

echo
echo "YUC Player is installed!"
echo "You can now launch it by double-clicking run.sh in:"
echo "  $APP_DIR"
echo
echo "If your file manager asks, choose 'Run' or 'Execute'."
