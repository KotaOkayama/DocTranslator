#!/bin/bash
set -e

echo "Starting DocTranslator..."

# 依存関係のチェックとインストール
echo "Checking dependencies..."
pip install --no-cache-dir et-xmlfile>=2.0.0
pip install --no-cache-dir openpyxl>=3.1.0
pip install --no-cache-dir pandas>=1.3.0

# インストールの確認
python3 -c "import openpyxl; print('openpyxl version:', openpyxl.__version__)"
python3 -c "import pandas; print('pandas version:', pandas.__version__)"

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
