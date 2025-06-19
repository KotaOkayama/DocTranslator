FROM python:3.11-slim

# Metadata
LABEL maintainer="KotaOkayama" \
      description="Document Translation Service" \
      version="1.0.0"

# System dependencies and fonts
RUN apt-get update && apt-get install -y \
    fonts-liberation \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-dejavu-core \
    libreoffice \
    libreoffice-writer \
    libreoffice-java-common \
    curl \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

    # ボリュームのマウントポイントを作成
VOLUME ["/app/uploads", "/app/downloads", "/app/logs"]

# Locale settings
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    sed -i '/ja_JP.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p uploads downloads logs cache && \
    chmod 755 uploads downloads logs cache

# Create non-root user first
RUN useradd --create-home --shell /bin/bash app

# Create and set permissions for .env file
# RUN touch .env && \
#    chmod 666 .env && \
#    chown app:app .env

# Copy application code
COPY . .
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBUG=false \
    LOG_LEVEL=INFO \
    MAX_FILE_SIZE=100MB \
    UPLOAD_TIMEOUT=300

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
