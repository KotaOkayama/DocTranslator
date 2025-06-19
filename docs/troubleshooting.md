# Troubleshooting Guide

## 🔍 Common Issues and Solutions

### 1. Installation Issues

#### Docker Related Issues

##### Docker Daemon Not Running
```
Error: Cannot connect to the Docker daemon
```

**Solution**:
1. Check Docker Desktop status
```bash
# Check Docker service
docker info

# Start Docker Desktop
# macOS: Open Docker Desktop application
# Windows: Start Docker Desktop from system tray
# Linux: sudo systemctl start docker
```

##### Port Conflicts
```
Error: Ports are not available: listen tcp 0.0.0.0:8000: bind: address already in use
```

**Solution**:
1. Find process using the port
```bash
# macOS/Linux
lsof -i :8000

# Windows
netstat -ano | findstr :8000
```

2. Stop the process or change port in docker-compose.yml
```yaml
ports:
  - "8001:8000"  # Change to different port
```

### 2. API Key Issues

#### Invalid API Key
```
Error: API key is not valid
```

**Solution**:
1. Check .env file configuration
```bash
cat .env
# Verify GENAI_HUB_API_KEY is set correctly
```

2. Verify API key in GenAI Hub
3. Try regenerating API key

#### Missing API Key
```
Error: API key is not set
```

**Solution**:
1. Copy example environment file
```bash
cp .env.example .env
```

2. Add your API key to .env
```bash
echo "GENAI_HUB_API_KEY=your_api_key_here" >> .env
```

3. Restart application
```bash
docker-compose restart
```

### 3. PDF Conversion Issues

#### LibreOffice Not Found
```
Error: LibreOffice command not found
```

**Solution**:
1. Install LibreOffice in container
```dockerfile
# Add to Dockerfile
RUN apt-get update && apt-get install -y \
    libreoffice \
    libreoffice-writer \
    libreoffice-java-common
```

2. Rebuild container
```bash
docker-compose build --no-cache
```

#### PDF Conversion Fails
```
Error: Failed to convert PDF
```

**Solution**:
1. Check file permissions
```bash
# Set correct permissions
chmod 755 uploads downloads
```

2. Check available memory
```bash
# Check Docker resources
docker stats
```

3. Increase container memory limit
```yaml
# In docker-compose.yml
services:
  document-translator:
    deploy:
      resources:
        limits:
          memory: 4G
```

### 4. Translation Issues

#### Translation Timeout
```
Error: Translation request timed out
```

**Solution**:
1. Check network connection
2. Increase timeout settings
```python
# In app/core/translator.py
TRANSLATION_CONFIG = {
    'api_timeout': 60,  # Increase timeout
    'retry_count': 3    # Increase retries
}
```

#### Japanese Text Display Issues
```
Error: Japanese text appears as dots or squares
```

**Solution**:
1. Install Japanese fonts
```dockerfile
# Add to Dockerfile
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    fonts-noto-cjk-extra
```

2. Rebuild container
```bash
docker-compose build --no-cache
```

### 5. Performance Issues

#### Slow Response Times
**Solution**:
1. Check resource usage
```bash
# Monitor container resources
docker stats

# Check logs for slow operations
docker-compose logs -f
```

2. Optimize container resources
```yaml
# In docker-compose.yml
services:
  document-translator:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

#### Memory Issues
**Solution**:
1. Clear temporary files
```bash
# Remove temporary files
find uploads/ -type f -mtime +1 -delete
find downloads/ -type f -mtime +1 -delete
```

2. Clear translation cache
```bash
rm translation_cache.json
```

## 🔧 Debugging Tools

### 1. Log Analysis

#### Application Logs
```bash
# View application logs
tail -f logs/app.log

# Filter error logs
grep ERROR logs/app.log

# View last 100 lines
tail -n 100 logs/app.log
```

#### Docker Logs
```bash
# View container logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f document-translator

# View last hour's logs
docker-compose logs --since 1h
```

### 2. System Status

#### Container Status
```bash
# List containers
docker-compose ps

# Container details
docker inspect document-translator

# Resource usage
docker stats
```

#### Network Status
```bash
# Check network connections
netstat -tulpn

# Test API endpoint
curl http://localhost:8000/health
```

### 3. Debug Mode

#### Enable Debug Logging
```bash
# Set debug mode in .env
DEBUG=true
LOG_LEVEL=DEBUG

# Restart application
docker-compose restart
```

#### Interactive Debugging
```bash
# Access container shell
docker-compose exec document-translator bash

# Install debugging tools
pip install ipdb

# Debug Python code
python -m ipdb app/main.py
```

## 🔄 Recovery Procedures

### 1. Data Recovery

#### Backup Current State
```bash
# Backup data directories
tar -czf backup.tar.gz uploads/ downloads/

# Backup configuration
cp .env .env.backup
```

#### Restore from Backup
```bash
# Restore data
tar -xzf backup.tar.gz

# Restore configuration
cp .env.backup .env
```

### 2. Application Reset

#### Soft Reset
```bash
# Stop containers
docker-compose down

# Remove temporary files
rm -rf uploads/* downloads/*

# Restart containers
docker-compose up -d
```

#### Hard Reset
```bash
# Stop and remove everything
docker-compose down -v

# Remove all data
rm -rf uploads/* downloads/* logs/*
rm translation_cache.json

# Rebuild and restart
docker-compose up --build -d
```

### 3. Cache Clear

#### Clear Translation Cache
```bash
# Remove cache file
rm translation_cache.json

# Restart application
docker-compose restart
```

#### Clear Temporary Files
```bash
# Remove uploaded files
rm -rf uploads/*

# Remove downloaded files
rm -rf downloads/*

# Remove log files
rm -rf logs/*
```

## 📊 Monitoring

### 1. Resource Monitoring

#### System Resources
```bash
# CPU and Memory usage
top

# Disk usage
df -h

# Memory info
free -m
```

#### Docker Resources
```bash
# Container stats
docker stats

# Container processes
docker top document-translator
```

### 2. Log Monitoring

#### Real-time Monitoring
```bash
# Monitor all logs
tail -f logs/*.log

# Monitor error logs
tail -f logs/error.log

# Monitor access logs
tail -f logs/access.log
```

#### Log Analysis
```bash
# Count errors
grep ERROR logs/app.log | wc -l

# Find specific errors
grep "Translation failed" logs/app.log

# Analyze response times
grep "Response time" logs/app.log | awk '{print $NF}' | sort -n
```

## 🔍 Diagnostic Commands

### 1. System Information

#### Version Information
```bash
# Python version
python --version

# Docker version
docker --version

# LibreOffice version
libreoffice --version
```

#### Configuration Check
```bash
# Environment variables
env | grep GENAI

# Docker configuration
docker info

# Application configuration
cat .env
```

### 2. Network Diagnostics

#### Port Check
```bash
# Check port usage
lsof -i :8000

# Test connection
nc -zv localhost 8000

# Check HTTP response
curl -I http://localhost:8000
```

#### DNS Check
```bash
# DNS resolution
nslookup genaihub.viper.eyrie.cloud

# Network route
traceroute genaihub.viper.eyrie.cloud
```

## 📞 Support Information

### 1. Log Collection

#### Collect All Logs
```bash
# Create log archive
tar -czf support_logs.tar.gz logs/

# Include configuration
cp .env support_logs/
```

#### System Information
```bash
# System details
uname -a
docker version
python --version
```

### 2. Contact Information

- Technical Support: tech-support@example.com
- Emergency Contact: emergency@example.com
- Working Hours: 9:00-17:00 JST

### 3. Reporting Issues

Please include:
1. Error message and logs
2. Steps to reproduce
3. System information
4. Configuration details
```