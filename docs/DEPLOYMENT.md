# Deployment Guide - Drilling Optimization

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5004

CMD ["python", "app.py"]
```

### Build and Run
```bash
docker build -t drilling-optimization .
docker run -p 5004:5004 drilling-optimization
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5004:5004"
    environment:
      - FLASK_DEBUG=0
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_DEBUG | Enable debug mode | 1 |
| PORT | Server port | 5004 |
| HOST | Server host | 0.0.0.0 |

## Manual Deployment

### Prerequisites
- Python 3.8+
- pip

### Steps
```bash
git clone https://github.com/kcabrera83/drilling-optimization.git
cd drilling-optimization
pip install -r requirements.txt
python train.py
python app.py
```

## Production Considerations

### Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5004 app:app
```

### Security
- Set `DEBUG=False` in production
- Use HTTPS with a reverse proxy
- Validate drilling parameter ranges
- Add rate limiting on optimization endpoint

### Monitoring
- Monitor `/api/health` for uptime
- Log optimization requests and results
- Track ROP predictions vs actuals for model drift

### Performance
- Optimization endpoint performs grid search (60 combinations per request)
- Consider caching results for repeated formations
- Pre-load models at startup

## API Self-Documentation
Access OpenAPI docs at: `http://localhost:5004/api/docs`
