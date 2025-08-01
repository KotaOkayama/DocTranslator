# Production environment Dockerfile
FROM python:3.11-slim

# 証明書の設定を最初に行う
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# 証明書の完全な再設定
RUN apt-get update && \
    apt-get install -y --reinstall --no-install-recommends ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates --fresh && \
    c_rehash /etc/ssl/certs

# 証明書パスを環境変数で明示的に設定
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_DIR=/etc/ssl/certs

# 本番環境用のシステムパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip設定で証明書を明示的に指定
RUN mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "cert = /etc/ssl/certs/ca-certificates.crt" >> ~/.pip/pip.conf && \
    echo "trusted-host = pypi.org" >> ~/.pip/pip.conf && \
    echo "               pypi.python.org" >> ~/.pip/pip.conf && \
    echo "               files.pythonhosted.org" >> ~/.pip/pip.conf

# LibreOffice設定ディレクトリを作成
RUN mkdir -p /root/.config/libreoffice/4/user \
    && mkdir -p /tmp/libreoffice \
    && chmod 777 /tmp/libreoffice

# Pythonの依存関係をインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Excel関連の依存関係を先にインストール
RUN pip install --no-cache-dir \
    et-xmlfile>=2.0.0 && \
    pip install --no-cache-dir \
    openpyxl>=3.1.0 && \
    pip install --no-cache-dir \
    pandas>=1.3.0

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
    pdfplumber>=0.9.0 \
    reportlab>=3.6.0

# PyMuPDFを事前ビルド済みホイールでインストール
RUN pip install --no-cache-dir --only-binary=:all: PyMuPDF>=1.23.0 || \
    pip install --no-cache-dir --no-deps PyMuPDF>=1.23.0 || \
    echo "PyMuPDF installation failed, will use alternative PDF processing"

# pdf2docxをインストール（PyMuPDFが利用可能な場合のみ）
RUN python3 -c "import fitz" && \
    pip install --no-cache-dir pdf2docx>=0.5.6 || \
    echo "pdf2docx installation skipped due to PyMuPDF unavailability"

# docx2pdfをインストール
RUN pip install --no-cache-dir comtypes>=1.1.0 && \
    pip install --no-cache-dir docx2pdf>=0.1.8 || \
    echo "docx2pdf installation failed, will use LibreOffice fallback"

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

# アプリケーションコードのコピー
COPY . .

# 必要なディレクトリの作成と権限設定
RUN chmod -R 755 /usr/local/lib/python3.11/site-packages && \
    mkdir -p /app/uploads /app/downloads /app/logs /app/cache && \
    chmod -R 755 /app/uploads /app/downloads /app/logs /app/cache

# 本番環境用の環境変数設定
ENV DISPLAY=:99 \
    HOME=/tmp \
    TMPDIR=/tmp \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PRODUCTION=true

EXPOSE 8000

# 本番環境用の起動スクリプトを作成
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting DocTranslator in Production Mode..."\n\
echo "Starting Xvfb..."\n\
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX &\n\
sleep 3\n\
echo "Testing LibreOffice..."\n\
libreoffice --headless --version || echo "LibreOffice test failed"\n\
echo "Testing PyMuPDF..."\n\
python3 -c "import fitz; print(\"PyMuPDF (fitz) is available\")" || echo "PyMuPDF test failed"\n\
echo "Testing pdf2docx..."\n\
python3 -c "import pdf2docx; print(\"pdf2docx is available\")" || echo "pdf2docx test failed"\n\
echo "Testing docx2pdf..."\n\
python3 -c "import docx2pdf; print(\"docx2pdf is available\")" || echo "docx2pdf test failed"\n\
echo "Starting application in production mode..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4\n\
' > /app/start.sh && chmod +x /app/start.sh

# 本番環境用のヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
