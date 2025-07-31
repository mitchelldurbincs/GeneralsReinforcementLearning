# Docker Setup for Generals RL Project

This document provides comprehensive Docker containerization setup for the Generals.io RL training infrastructure.

## Container Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ Game Server │  │ RL Trainer  │  │ Experience Buffer │   │
│  │   (Go)      │  │  (Python)   │  │    (Go/Redis)     │   │
│  └─────────────┘  └─────────────┘  └───────────────────┘   │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │
│  │ Agent Pool  │  │ Monitoring  │  │   Model Server    │   │
│  │  (Python)   │  │ (Grafana)   │  │    (Python)       │   │
│  └─────────────┘  └─────────────┘  └───────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Container Definitions

### 1. Game Server Container

**File**: `docker/game-server/Dockerfile`

```dockerfile
# Multi-stage build for optimal size
FROM golang:1.24-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make

WORKDIR /build

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build with optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s" \
    -o game-server \
    ./cmd/game_server

# Runtime stage
FROM alpine:3.21

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN adduser -D -g '' appuser

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/game-server .
COPY --from=builder /build/configs ./configs

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/app/game-server", "health"] || exit 1

ENTRYPOINT ["/app/game-server"]
```

### 2. Python RL Agent Container

**File**: `docker/rl-agent/Dockerfile`

```dockerfile
# Use official Python image with CUDA support for GPU training
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04 AS base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY python/requirements.txt .
COPY python/requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy Python package
COPY python/ ./python/
COPY proto/ ./proto/

# Install the package
RUN pip install -e ./python

# Create non-root user
RUN useradd -m -s /bin/bash agent
USER agent

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/python:$PYTHONPATH

# Default command
CMD ["python", "-m", "generals_agent.agent_runner"]
```

## Docker Compose Configurations

### Development Setup

**File**: `docker-compose.dev.yml`

```yaml
version: '3.8'

services:
  game-server:
    build:
      context: .
      dockerfile: docker/game-server/Dockerfile
    ports:
      - "50051:50051"
    environment:
      - LOG_LEVEL=debug
      - ENVIRONMENT=development
    volumes:
      - ./configs:/app/configs:ro
    networks:
      - generals-net

  experience-buffer:
    build:
      context: .
      dockerfile: docker/experience-buffer/Dockerfile
    ports:
      - "50052:50052"
    environment:
      - BUFFER_SIZE=100000
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    networks:
      - generals-net

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - generals-net

  rl-agent:
    build:
      context: .
      dockerfile: docker/rl-agent/Dockerfile
    environment:
      - GAME_SERVER=game-server:50051
      - EXPERIENCE_BUFFER=experience-buffer:50052
      - PYTHONUNBUFFERED=1
    volumes:
      - ./python:/app/python:ro
      - ./models:/app/models
    depends_on:
      - game-server
      - experience-buffer
    networks:
      - generals-net
    deploy:
      replicas: 4  # Multiple agents for self-play

  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - generals-net

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - generals-net

volumes:
  redis-data:
  grafana-data:
  prometheus-data:

networks:
  generals-net:
    driver: bridge
```

### Production Setup

**File**: `docker-compose.prod.yml`

```yaml
version: '3.8'

services:
  game-server:
    image: ${ECR_REGISTRY}/generals-game-server:${VERSION}
    ports:
      - "50051:50051"
    environment:
      - LOG_LEVEL=info
      - ENVIRONMENT=production
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    networks:
      - generals-net

  rl-trainer:
    image: ${ECR_REGISTRY}/generals-rl-trainer:${VERSION}
    environment:
      - GAME_SERVER=game-server:50051
      - EXPERIENCE_BUFFER=experience-buffer:50052
      - S3_BUCKET=${S3_BUCKET}
      - MODEL_CHECKPOINT_DIR=s3://${S3_BUCKET}/checkpoints
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '8'
          memory: 32G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - generals-net

networks:
  generals-net:
    external: true
```

## Development Workflow

### Local Development with Hot Reload

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Watch logs
docker-compose -f docker-compose.dev.yml logs -f game-server

# Run tests in container
docker-compose -f docker-compose.dev.yml exec game-server go test ./...

# Access development shell
docker-compose -f docker-compose.dev.yml exec dev bash
```

### Running RL Training Locally

```bash
# Start training environment
docker-compose -f docker-compose.dev.yml up -d game-server experience-buffer redis

# Scale up agents for self-play
docker-compose -f docker-compose.dev.yml up -d --scale rl-agent=8

# Monitor training
docker-compose -f docker-compose.dev.yml logs -f rl-agent
```

## Best Practices

### 1. Image Optimization
- Use multi-stage builds to minimize final image size
- Order Dockerfile commands to maximize layer caching
- Use specific version tags instead of `latest`
- Remove unnecessary files and packages

### 2. Security
- Run containers as non-root users
- Use read-only root filesystems where possible
- Scan images for vulnerabilities
- Use secrets management for sensitive data

### 3. Resource Management
- Set appropriate resource limits
- Use health checks for all services
- Implement graceful shutdown handling
- Monitor resource usage

### 4. Networking
- Use custom networks instead of default bridge
- Implement service discovery
- Use internal networks for backend services
- Expose only necessary ports

## Additional Files

### .dockerignore
```
# Git
.git
.gitignore

# Go
vendor/
*.test
*.out

# Python
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/

# IDEs
.vscode/
.idea/

# OS
.DS_Store

# Project specific
/models/
/logs/
/data/
```

### Makefile Docker Commands
```makefile
# Docker commands
.PHONY: docker-build docker-up docker-down docker-logs

docker-build:
	docker-compose -f docker-compose.dev.yml build

docker-up:
	docker-compose -f docker-compose.dev.yml up -d

docker-down:
	docker-compose -f docker-compose.dev.yml down

docker-logs:
	docker-compose -f docker-compose.dev.yml logs -f

docker-test:
	docker-compose -f docker-compose.dev.yml run --rm game-server go test ./...
	docker-compose -f docker-compose.dev.yml run --rm rl-agent pytest
```

## Migration Path

### Phase 1: Local Development (Immediate)
1. Create Docker files for each service
2. Set up docker-compose for local development
3. Test multi-container setup locally
4. Add .dockerignore files

### Phase 2: CI/CD Integration (Week 1-2)
1. Set up GitHub Actions for automated builds
2. Create ECR repositories
3. Implement automated testing in containers
4. Add security scanning

### Phase 3: Cloud Deployment (Week 3-4)
1. Deploy to ECS or EKS
2. Set up service discovery
3. Implement monitoring and logging
4. Configure auto-scaling

## Key Benefits

- **Modularity**: Each service in its own container
- **Scalability**: Easy to scale individual components
- **Portability**: Consistent environments across dev/prod
- **Efficiency**: Optimized images and resource usage
- **Observability**: Built-in monitoring and logging