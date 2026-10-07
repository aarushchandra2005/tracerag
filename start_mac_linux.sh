#!/usr/bin/env bash
# Start TraceRAG. The first run creates a private Python environment in .venv
# and installs the packages (a few minutes).
set -e
cd "$(dirname "$0")"

if [ ! -f .venv/installed.ok ]; then
  # Python 3.10 to 3.14 (PyTorch has no packages for newer versions yet). Override with PYTHON=/path/to/python.
  supported() { "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)' 2>/dev/null; }
  PY="${PYTHON:-}"
  if [ -z "$PY" ]; then
    for c in python3.13 python3.12 python3.14 python3.11 python3.10 python3 python; do
      if command -v "$c" >/dev/null 2>&1 && supported "$c"; then PY="$c"; break; fi
    done
  fi
  if [ -z "$PY" ] || ! supported "$PY"; then
    echo "TraceRAG needs Python 3.10 to 3.14 (3.13 is a good choice): https://www.python.org/downloads/"
    exit 1
  fi
  echo "Setting up TraceRAG for the first time with $PY. This takes a few minutes."
  "$PY" -m venv --clear .venv || { echo "Could not create .venv. On Debian or Ubuntu: sudo apt install python3-venv"; exit 1; }
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "Installing the packages failed. Read the messages above, fix the problem, then run this again."; exit 1; }
  touch .venv/installed.ok
fi
exec .venv/bin/python run_app.py "$@"
