#!/bin/bash
set -e

echo "Starting DocTranslator..."

# 仮想ディスプレイを開始
echo "Starting Xvfb..."
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX &
sleep 3

# LibreOfficeのテスト
echo "Testing LibreOffice..."
libreoffice --headless --version || echo "LibreOffice test failed"

# アプリケーションを開始
echo "Starting application..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
