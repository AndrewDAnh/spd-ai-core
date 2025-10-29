# Deployment Guide

## Local Development Setup

### Prerequisites
- Python 3.11
- pip
- Virtual environment (venv)

### Quick Start

1. **Clone and Navigate**
```bash
cd spd-mvp
```

2. **Setup Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
cp .env.example .env
# Edit .env if needed
```

5. **Run the Server**
```bash
# Option 1: Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using the startup script
chmod +x run.sh
./run.sh
```

6. **Access the API**
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing

Run the test script to verify all endpoints:
```bash
# Make sure the API is running first
python test_api.py
```

## Docker Deployment (Optional)

### Dockerfile

Create a `Dockerfile`:

```dockerfile
FROM python:3.11

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY .env .env

# Create data directory
RUN mkdir -p data

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t spd-mvp:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --name spd-mvp \
  spd-mvp:latest

# View logs
docker logs -f spd-mvp
```

## Production Deployment

### Environment Variables

For production, set these environment variables:

```bash
APP_NAME=SPD-MVP
APP_VERSION=1.0.0
DEBUG=False
DATABASE_URL=sqlite:///./data/predictions.db
FAILURE_THRESHOLD=30.0
DRIFT_THRESHOLD=0.2
PSI_THRESHOLD=0.2
OUTLIER_SENSITIVITY=medium
```

### Using Gunicorn (Production ASGI Server)

1. **Install Gunicorn**
```bash
pip install gunicorn
```

2. **Run with Gunicorn**
```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Nginx Reverse Proxy

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Systemd Service

Create `/etc/systemd/system/spd-mvp.service`:

```ini
[Unit]
Description=SPD-MVP API Service
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/spd-mvp
Environment="PATH=/opt/spd-mvp/venv/bin"
ExecStart=/opt/spd-mvp/venv/bin/gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable spd-mvp
sudo systemctl start spd-mvp
sudo systemctl status spd-mvp
```

## Cloud Deployment

### AWS EC2

1. Launch EC2 instance (Ubuntu 20.04+)
2. Install Python and dependencies
3. Clone repository
4. Setup as systemd service
5. Configure security group (port 8000)
6. Optional: Setup ALB for load balancing

### AWS Lambda + API Gateway

For serverless deployment:
1. Use Mangum adapter for FastAPI
2. Package application as Lambda function
3. Configure API Gateway
4. Use RDS or DynamoDB instead of SQLite

### Google Cloud Run

```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/spd-mvp

# Deploy
gcloud run deploy spd-mvp \
  --image gcr.io/PROJECT_ID/spd-mvp \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Monitoring

### Logging

The application logs to stdout. In production:

```bash
# View logs (systemd)
journalctl -u spd-mvp -f

# View logs (Docker)
docker logs -f spd-mvp
```

### Health Checks

Configure health check monitoring:
```bash
# Simple health check
curl http://localhost:8000/api/v1/health

# Add to monitoring tool (Uptime Robot, Pingdom, etc.)
```

### Metrics

For production, add:
- Prometheus metrics endpoint
- APM tool (DataDog, New Relic)
- Error tracking (Sentry)

## Scaling

### Horizontal Scaling

1. Deploy multiple instances
2. Use load balancer (Nginx, AWS ALB)
3. Share SQLite via network storage (or migrate to PostgreSQL)

### Vertical Scaling

Adjust worker count based on CPU cores:
```bash
workers = (2 x CPU_CORES) + 1
```

## Backup

### Database Backup

```bash
# Backup SQLite database
cp data/predictions.db data/predictions.db.backup.$(date +%Y%m%d)

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/spd-mvp"
mkdir -p $BACKUP_DIR
cp data/predictions.db $BACKUP_DIR/predictions.db.$(date +%Y%m%d-%H%M%S)
# Keep only last 30 days
find $BACKUP_DIR -name "predictions.db.*" -mtime +30 -delete
```

## Troubleshooting

### Common Issues

**Port already in use**
```bash
lsof -i :8000
kill -9 <PID>
```

**Database locked**
```bash
# SQLite doesn't support high concurrency
# Consider migrating to PostgreSQL for production
```

**Import errors**
```bash
# Ensure you're in the correct directory and venv is activated
cd /path/to/spd-mvp
source venv/bin/activate
```

## Security Checklist

- [ ] Disable DEBUG mode in production
- [ ] Add authentication (JWT, OAuth2)
- [ ] Setup HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Add rate limiting
- [ ] Setup firewall rules
- [ ] Regular security updates
- [ ] Database encryption
- [ ] Input validation
- [ ] API key management

## Performance Tuning

1. **Database Optimization**
   - Add indexes on frequently queried columns
   - Use connection pooling
   - Consider PostgreSQL for production

2. **Caching**
   - Cache reference baselines
   - Cache model predictions
   - Use Redis for distributed caching

3. **Request Optimization**
   - Implement pagination
   - Add request size limits
   - Use async database operations

4. **Model Loading**
   - Load model once at startup
   - Use model caching
   - Consider model serving platforms (TorchServe, TensorFlow Serving)

