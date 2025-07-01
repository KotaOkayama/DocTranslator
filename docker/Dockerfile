FROM python:3.11-slim

# システムパッケージの更新とLibreOffice関連の完全インストール
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    gcc \
    g++ \
    xvfb \
    libreoffice \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    libreoffice-draw \
    libreoffice-java-common \
    libreoffice-l10n-en-us \
    libreoffice-help-en-us \
    default-jre \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-liberation2 \
    fonts-crosextra-carlito \
    fonts-crosextra-caladea \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# LibreOffice設定ディレクトリを作成
RUN mkdir -p /root/.config/libreoffice/4/user \
    && mkdir -p /tmp/libreoffice \
    && chmod 777 /tmp/libreoffice

WORKDIR /app

# Pythonの依存関係をインストール
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Excel関連の依存関係を先にインストール
RUN pip install --no-cache-dir et-xmlfile>=2.0.0 && \
    pip install --no-cache-dir openpyxl>=3.1.0 && \
    pip install --no-cache-dir pandas>=1.3.0

# Excel関連パッケージのインストール確認
RUN python3 -c "import openpyxl, et_xmlfile, pandas; \
    print('Installed versions:', \
    'openpyxl:', openpyxl.__version__, \
    'pandas:', pandas.__version__, \
    'et_xmlfile is installed')" || true

# 基本的な依存関係をインストール
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    starlette==0.27.0 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    websockets==12.0 \
    python-dotenv==1.0.0

# データ処理関連のパッケージをインストール
RUN pip install --no-cache-dir python-pptx>=0.6.21 && \
    pip install --no-cache-dir python-docx>=0.8.11

# PDF処理関連パッケージをインストール
RUN pip install --no-cache-dir \
    PyPDF2>=3.0.0 \
    pdfplumber>=0.9.0

# その他のユーティリティをインストール
RUN pip install --no-cache-dir \
    requests>=2.26.0 \
    tqdm>=4.62.0 \
    Pillow>=9.0.0 \
    python-magic>=0.4.27 \
    nltk>=3.8 \
    cryptography>=41.0.0 \
    prometheus-client>=0.17.0 \
    fonttools>=4.38.0 \
    httpx==0.24.1

# 開発用依存関係をインストール
RUN pip install --no-cache-dir -r requirements-dev.txt

# アプリケーションコードのコピー
COPY . .

# 必要なディレクトリの作成と権限設定
RUN chmod -R 777 /usr/local/lib/python3.11/site-packages && \
    mkdir -p /app/uploads /app/downloads /app/logs /app/cache && \
    chmod -R 777 /app/uploads /app/downloads /app/logs /app/cache

# 環境変数設定
ENV DISPLAY=:99 \
    HOME=/tmp \
    TMPDIR=/tmp \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# 起動スクリプトを作成
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting DocTranslator..."\n\
echo "Starting Xvfb..."\n\
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX &\n\
sleep 3\n\
echo "Testing LibreOffice..."\n\
libreoffice --headless --version || echo "LibreOffice test failed"\n\
echo "Starting application..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload\n\
' > /app/start.sh && chmod +x /app/start.sh

# ヘルスチェックの追加
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
