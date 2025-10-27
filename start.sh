#!/bin/bash
set -e

echo "Starting DocTranslator..."

# Check and install dependencies
echo "Checking dependencies..."
pip install --no-cache-dir et-xmlfile>=2.0.0
pip install --no-cache-dir openpyxl>=3.1.0
pip install --no-cache-dir pandas>=1.3.0

# Verify installation
python3 -c "import openpyxl; print('openpyxl version:', openpyxl.__version__)"
python3 -c "import pandas; print('pandas version:', pandas.__version__)"

# Start virtual display (check if already running)
echo "Starting Xvfb..."
if pgrep -x "Xvfb" > /dev/null; then
    echo "Xvfb is already running"
else
    # Remove stale lock file if exists
    rm -f /tmp/.X99-lock
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX &
    sleep 3
    echo "Xvfb started"
fi

# Test LibreOffice
echo "Testing LibreOffice..."
libreoffice --headless --version || echo "LibreOffice test failed"

# Start application
echo "Starting application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
