# 🚀 Kangro Capital Deployment Guide

This guide provides comprehensive instructions for deploying the Kangro Capital platform in various environments.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for API access

### API Keys Required
- **Polygon.io**: For real-time market data
- **Tavily**: For market news and sentiment analysis
- **OpenAI**: For AI-powered insights (optional)

## 🏠 Local Development Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/kaljuvee/kangro-capital.git
cd kangro-capital

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements_core.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

Required environment variables:
```env
POLYGON_API_KEY=your_polygon_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

### 3. Run Application

```bash
# Start the Streamlit application
streamlit run streamlit_app.py

# Application will be available at:
# http://localhost:8501
```

### 4. Verify Installation

```bash
# Run comprehensive tests
python test_comprehensive.py

# Expected output: 87.5% success rate or higher
```

## ☁️ Cloud Deployment

### Streamlit Cloud Deployment

1. **Fork the repository** to your GitHub account

2. **Visit Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Deploy the app**
   - Click "New app"
   - Select your forked repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

4. **Configure secrets**
   - In Streamlit Cloud dashboard, go to app settings
   - Add secrets in TOML format:
   ```toml
   POLYGON_API_KEY = "your_key_here"
   TAVILY_API_KEY = "your_key_here"
   OPENAI_API_KEY = "your_key_here"
   OPENAI_API_BASE = "https://api.openai.com/v1"
   ```

### Heroku Deployment

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Ubuntu
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Create Heroku app**
   ```bash
   heroku create your-app-name
   heroku config:set POLYGON_API_KEY=your_key_here
   heroku config:set TAVILY_API_KEY=your_key_here
   heroku config:set OPENAI_API_KEY=your_key_here
   ```

3. **Create Procfile**
   ```bash
   echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### AWS EC2 Deployment

1. **Launch EC2 instance**
   - Choose Ubuntu 22.04 LTS
   - Instance type: t3.medium or larger
   - Configure security group to allow port 8501

2. **Connect and setup**
   ```bash
   # Connect to instance
   ssh -i your-key.pem ubuntu@your-instance-ip
   
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and pip
   sudo apt install python3.11 python3.11-venv python3-pip -y
   
   # Clone repository
   git clone https://github.com/kaljuvee/kangro-capital.git
   cd kangro-capital
   
   # Setup virtual environment
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements_core.txt
   ```

3. **Configure environment**
   ```bash
   # Create .env file
   nano .env
   # Add your API keys
   ```

4. **Run with systemd**
   ```bash
   # Create service file
   sudo nano /etc/systemd/system/kangro-capital.service
   ```
   
   Service file content:
   ```ini
   [Unit]
   Description=Kangro Capital Streamlit App
   After=network.target
   
   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/kangro-capital
   Environment=PATH=/home/ubuntu/kangro-capital/venv/bin
   ExecStart=/home/ubuntu/kangro-capital/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   ```bash
   # Enable and start service
   sudo systemctl enable kangro-capital
   sudo systemctl start kangro-capital
   sudo systemctl status kangro-capital
   ```

## 🐳 Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_core.txt .
RUN pip install --no-cache-dir -r requirements_core.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  kangro-capital:
    build: .
    ports:
      - "8501:8501"
    environment:
      - POLYGON_API_KEY=${POLYGON_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_API_BASE=${OPENAI_API_BASE}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Build and run

```bash
# Build image
docker build -t kangro-capital .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔧 Production Configuration

### Environment Variables

```env
# Production settings
DEBUG=False
LOG_LEVEL=INFO
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# API Configuration
POLYGON_API_KEY=your_production_key
TAVILY_API_KEY=your_production_key
OPENAI_API_KEY=your_production_key
OPENAI_API_BASE=https://api.openai.com/v1

# Performance settings
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200

# Security settings
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

### Performance Optimization

1. **Enable caching**
   ```python
   # Already implemented in the application
   @st.cache_resource
   def initialize_components():
       # Component initialization
   ```

2. **Configure resource limits**
   ```bash
   # For systemd service
   [Service]
   MemoryLimit=2G
   CPUQuota=200%
   ```

3. **Enable compression**
   ```bash
   # Nginx configuration
   gzip on;
   gzip_types text/plain application/json application/javascript text/css;
   ```

### Security Hardening

1. **Firewall configuration**
   ```bash
   # Ubuntu UFW
   sudo ufw allow 22/tcp
   sudo ufw allow 8501/tcp
   sudo ufw enable
   ```

2. **SSL/TLS setup with Nginx**
   ```nginx
   server {
       listen 443 ssl;
       server_name your-domain.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

3. **Environment security**
   ```bash
   # Secure .env file
   chmod 600 .env
   chown root:root .env
   ```

## 📊 Monitoring and Logging

### Application Monitoring

1. **Health checks**
   ```python
   # Built into Streamlit
   # Access: http://your-app/_stcore/health
   ```

2. **Custom monitoring**
   ```python
   import logging
   
   # Configure logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('kangro-capital.log'),
           logging.StreamHandler()
       ]
   )
   ```

3. **Performance monitoring**
   ```bash
   # System monitoring
   htop
   iotop
   netstat -tulpn
   ```

### Log Management

1. **Log rotation**
   ```bash
   # Create logrotate configuration
   sudo nano /etc/logrotate.d/kangro-capital
   ```
   
   ```
   /home/ubuntu/kangro-capital/*.log {
       daily
       missingok
       rotate 30
       compress
       delaycompress
       notifempty
       create 644 ubuntu ubuntu
   }
   ```

2. **Centralized logging** (optional)
   ```bash
   # Install and configure rsyslog for centralized logging
   sudo apt install rsyslog
   ```

## 🔄 Backup and Recovery

### Data Backup

```bash
# Create backup script
#!/bin/bash
BACKUP_DIR="/backup/kangro-capital"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup application data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Backup configuration
cp .env $BACKUP_DIR/env_$DATE.backup

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

1. **Application recovery**
   ```bash
   # Stop application
   sudo systemctl stop kangro-capital
   
   # Restore from backup
   tar -xzf backup/data_latest.tar.gz
   tar -xzf backup/models_latest.tar.gz
   
   # Restart application
   sudo systemctl start kangro-capital
   ```

2. **Database recovery** (if using external database)
   ```bash
   # Restore database from backup
   # Implementation depends on database type
   ```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port 8501
   lsof -i :8501
   
   # Kill process
   kill -9 <PID>
   
   # Or use different port
   streamlit run streamlit_app.py --server.port=8502
   ```

2. **Memory issues**
   ```bash
   # Check memory usage
   free -h
   
   # Check application memory
   ps aux | grep streamlit
   
   # Restart application
   sudo systemctl restart kangro-capital
   ```

3. **API connection issues**
   ```bash
   # Test API connectivity
   curl -H "Authorization: Bearer $POLYGON_API_KEY" \
        "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02"
   
   # Check environment variables
   echo $POLYGON_API_KEY
   ```

4. **SSL certificate issues**
   ```bash
   # Renew Let's Encrypt certificate
   sudo certbot renew
   
   # Restart nginx
   sudo systemctl restart nginx
   ```

### Performance Issues

1. **Slow loading**
   - Check internet connection
   - Verify API rate limits
   - Monitor system resources
   - Clear Streamlit cache

2. **High memory usage**
   - Reduce dataset size
   - Implement data pagination
   - Clear unused variables
   - Restart application

### Debug Mode

```bash
# Run in debug mode
export DEBUG=True
streamlit run streamlit_app.py --logger.level=debug
```

## 📞 Support

For deployment issues:
- **GitHub Issues**: Technical problems
- **Discussions**: General questions
- **Documentation**: Check existing guides
- **Community**: Stack Overflow with tag `kangro-capital`

## 📈 Scaling

### Horizontal Scaling

1. **Load balancer setup**
   ```nginx
   upstream kangro_backend {
       server 127.0.0.1:8501;
       server 127.0.0.1:8502;
       server 127.0.0.1:8503;
   }
   
   server {
       location / {
           proxy_pass http://kangro_backend;
       }
   }
   ```

2. **Multiple instances**
   ```bash
   # Run multiple instances on different ports
   streamlit run streamlit_app.py --server.port=8501 &
   streamlit run streamlit_app.py --server.port=8502 &
   streamlit run streamlit_app.py --server.port=8503 &
   ```

### Vertical Scaling

- Increase server resources (CPU, RAM)
- Optimize database queries
- Implement caching strategies
- Use CDN for static assets

---

This deployment guide covers most common scenarios. For specific requirements or issues, please refer to the troubleshooting section or create an issue on GitHub.

