# TeleChat Docker Quick Reference

## Quick Start
```bash
# 1. Download models to ./models/7B directory
# 2. Start services
docker-compose up -d

# 3. Access services
# API: http://localhost:8070/docs
# Web: http://localhost:8501
```

## Common Commands

### Start Services
```bash
docker-compose up -d          # Start in background
docker-compose up             # Start with logs
```

### View Logs
```bash
docker-compose logs -f        # Follow logs
docker-compose logs           # View all logs
```

### Stop Services
```bash
docker-compose down           # Stop and remove containers
docker-compose stop           # Stop containers only
```

### Restart Services
```bash
docker-compose restart        # Restart all services
```

### Check Status
```bash
docker-compose ps             # List containers
docker stats telechat-service # Resource usage
```

## Configuration

### Environment Variables (.env file)
```bash
MODEL_PATH=/app/models/7B     # Model location
CUDA_VISIBLE_DEVICES=0        # GPU selection
API_PORT=8070                 # API port
WEB_PORT=8501                 # Web port
```

### Custom Model Path
```bash
MODEL_PATH=/app/models/12B docker-compose up -d
```

### Multiple GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1 docker-compose up -d
```

### Custom Ports
```bash
API_PORT=8080 WEB_PORT=8502 docker-compose up -d
```

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :8070
sudo lsof -i :8501

# Change ports in .env or environment variable
API_PORT=8080 docker-compose up -d
```

### GPU Not Available
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# If failed, install nvidia-container-toolkit
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

### Container Won't Start
```bash
# Check logs
docker-compose logs telechat

# Check if models directory exists
ls -la ./models/7B

# Rebuild image
docker-compose build --no-cache
docker-compose up -d
```

### Out of Memory
```bash
# Use quantized model
MODEL_PATH=/app/models/7B-int4 docker-compose up -d

# Check available memory
free -h
nvidia-smi
```

## Development

### Code Hot Reload
The service directory is mounted as a volume, so code changes take effect after restarting:
```bash
docker-compose restart
```

### Interactive Shell
```bash
docker exec -it telechat-service bash
```

### Rebuild After Changes
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

## Maintenance

### Clean Up
```bash
# Remove containers and networks
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v

# Remove unused images
docker image prune -a
```

### Update Base Image
```bash
docker-compose pull
docker-compose up -d --build
```

## For More Information
- Full documentation: [DOCKER.md](./DOCKER.md)
- Deployment guide: [DEPLOYMENT.md](./DEPLOYMENT.md)
- Main README: [README.md](./README.md)
