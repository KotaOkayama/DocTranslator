# Deployment Guide

## 🚀 Deployment Options

### Docker Deployment (Recommended)

1. **Prerequisites**
- Docker Engine 24.0.0+
- Docker Compose 2.20.0+
- 4GB RAM minimum
- GenAI Hub API key

2. **Production Setup**
```bash
# Clone repository
git clone https://github.com/CS-Japan-SE/DocTranslator.git
cd DocTranslator

# Create environment file
cp .env.example .env

# Edit .env with production settings
cat << EOF > .env
GENAI_HUB_API_KEY=your_api_key_here
DEBUG=false
LOG_LEVEL=INFO
MAX_FILE_SIZE=100MB
UPLOAD_TIMEOUT=300
EOF

# Build and start containers
docker-compose up --build -d
```

3. **Directory Structure**
```
/opt/doctranslator/
├── app/
├── docker/
├── downloads/
├── uploads/
├── logs/
├── .env
└── docker-compose.yml
```

4. **File Permissions**
```bash
# Set correct permissions
sudo chown -R 1000:1000 downloads uploads logs
sudo chmod 755 downloads uploads logs
```

## 📊 Server Requirements

### Minimum Requirements
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB
- OS: Ubuntu 22.04 LTS or later

### Recommended Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB
- OS: Ubuntu 22.04 LTS or later

## 🔧 Configuration

### Environment Variables

```env
# Required Settings
GENAI_HUB_API_KEY=your_api_key_here

# Optional Settings
DEBUG=false
LOG_LEVEL=INFO
MAX_FILE_SIZE=100MB
UPLOAD_TIMEOUT=300

# Advanced Settings
WORKERS=4
KEEPALIVE=65
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  document-translator:
    build: .
    restart: unless-stopped
    environment:
      - GENAI_HUB_API_KEY=${GENAI_HUB_API_KEY}
      - DEBUG=false
      - LOG_LEVEL=INFO
    volumes:
      - ./uploads:/app/uploads
      - ./downloads:/app/downloads
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔒 Security Configuration

### SSL/TLS Setup

1. **Generate SSL Certificate**
```bash
mkdir -p docker/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/nginx/ssl/key.pem \
  -out docker/nginx/ssl/cert.pem
```

2. **Nginx Configuration**
```nginx
server {
    listen 443 ssl;
    server_name your_domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://document-translator:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Firewall Configuration

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow SSH
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

## 📈 Monitoring

### Health Checks

1. **Application Health**
```bash
# Check application status
curl http://localhost:8000/health

# Check container health
docker inspect --format "{{json .State.Health }}" document-translator
```

2. **Resource Monitoring**
```bash
# Monitor container resources
docker stats document-translator

# Check logs
docker-compose logs -f
```

### Logging

1. **Application Logs**
```bash
# View application logs
tail -f logs/app.log

# View error logs
tail -f logs/error.log
```

2. **Docker Logs**
```bash
# View container logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f document-translator
```

## 🔄 Maintenance

### Backup Procedures

1. **Data Backup**
```bash
# Create backup directory
mkdir -p backups

# Backup files
tar -czf backups/files_$(date +%Y%m%d).tar.gz uploads/ downloads/

# Backup configuration
cp .env backups/env_$(date +%Y%m%d).bak
```

2. **Backup Script**
```bash
#!/bin/bash
BACKUP_DIR="/opt/doctranslator/backups"
DATE=$(date +%Y%m%d)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup files
tar -czf "$BACKUP_DIR/files_$DATE.tar.gz" uploads/ downloads/

# Backup configuration
cp .env "$BACKUP_DIR/env_$DATE.bak"

# Remove backups older than 30 days
find "$BACKUP_DIR" -type f -mtime +30 -delete
```

### Updates

1. **Application Update**
```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose up --build -d
```

2. **System Updates**
```bash
# Update system packages
sudo apt update
sudo apt upgrade

# Update Docker images
docker-compose pull
```

## 🔄 Scaling

### Horizontal Scaling

1. **Docker Swarm Setup**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml doctranslator
```

2. **Load Balancer Configuration**
```nginx
upstream doctranslator {
    server document-translator:8000;
    server document-translator:8001;
    server document-translator:8002;
}
```

### Resource Scaling

```yaml
services:
  document-translator:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## 🚨 Troubleshooting

### Common Issues

1. **Container Won't Start**
```bash
# Check logs
docker-compose logs document-translator

# Check container status
docker ps -a

# Check resource usage
docker stats
```

2. **Performance Issues**
```bash
# Monitor resource usage
top
free -m
df -h

# Check application logs
tail -f logs/app.log
```

### Recovery Procedures

1. **Service Recovery**
```bash
# Restart service
docker-compose restart document-translator

# Rebuild service
docker-compose up --build -d document-translator
```

2. **Data Recovery**
```bash
# Restore from backup
tar -xzf backups/files_YYYYMMDD.tar.gz
cp backups/env_YYYYMMDD.bak .env
```

## 📝 Post-Deployment Checklist

### Security
- [ ] SSL/TLS configured
- [ ] Firewall enabled
- [ ] Secure file permissions
- [ ] Environment variables set
- [ ] Backup system configured

### Monitoring
- [ ] Health checks enabled
- [ ] Logging configured
- [ ] Resource monitoring setup
- [ ] Alert system configured

### Documentation
- [ ] Deployment documented
- [ ] Backup procedures documented
- [ ] Recovery procedures documented
- [ ] Contact information updated

## 📞 Support

### Contact Information
- Technical Support: tech-support@example.com
- Emergency Contact: emergency@example.com

### Useful Commands
```bash
# Check application status
docker-compose ps

# View logs
docker-compose logs -f

# Restart application
docker-compose restart

# Update application
git pull && docker-compose up --build -d
```

## 🔄 Rollback Procedures

### Quick Rollback
```bash
# Stop current version
docker-compose down

# Checkout previous version
git checkout <previous-tag>

# Start previous version
docker-compose up -d
```

### Full Rollback
```bash
# Backup current data
./scripts/backup.sh

# Restore previous version
git checkout <previous-tag>
tar -xzf backups/files_YYYYMMDD.tar.gz
cp backups/env_YYYYMMDD.bak .env
docker-compose up --build -d
```

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
scrape_configs:
  - job_name: 'doctranslator'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "id": null,
    "title": "DocTranslator Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph"
      },
      {
        "title": "Response Time",
        "type": "graph"
      }
    ]
  }
}
```

## 🔐 Security Hardening

### System Hardening
```bash
# Update system
sudo apt update && sudo apt upgrade

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable

# Secure shared memory
echo "tmpfs     /run/shm     tmpfs     defaults,noexec,nosuid     0     0" >> /etc/fstab
```

### Docker Security
```bash
# Create docker group
sudo groupadd docker
sudo usermod -aG docker $USER

# Configure docker daemon
cat << EOF > /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "userns-remap": "default"
}
EOF
```

## 📈 Performance Tuning

### System Settings
```bash
# Increase file descriptors
echo "* soft nofile 65535" >> /etc/security/limits.conf
echo "* hard nofile 65535" >> /etc/security/limits.conf

# Optimize network settings
cat << EOF >> /etc/sysctl.conf
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 30
EOF
```

### Application Settings
```yaml
services:
  document-translator:
    environment:
      - WORKERS=4
      - KEEPALIVE=65
      - MAX_REQUESTS=1000
      - MAX_REQUESTS_JITTER=50
```
```