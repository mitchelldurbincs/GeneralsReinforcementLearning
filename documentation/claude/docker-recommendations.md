# Docker Recommendations for Generals RL Project

## Executive Summary

This document provides comprehensive Docker containerization recommendations for the Generals.io RL training infrastructure. The strategy focuses on creating modular, scalable containers that support both local development and cloud deployment for distributed RL training.

## Current State Analysis

### Existing Setup
- Single Dockerfile in `/deploy/Dockerfile` for game server only
- Builds a lightweight Alpine-based container for the game server
- No containerization for Python agents or RL training components
- No docker-compose setup for local development
- No multi-service orchestration

### Key Gaps
1. No Python environment containerization
2. Missing development vs production configurations
3. No GPU support for RL training
4. Lack of service orchestration
5. No environment-specific configurations

## Recommended Container Architecture

### Service Separation Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │ Game Server │  │ RL Trainer  │  │ Experience Buffer │  │
│  │   (Go)      │  │  (Python)   │  │    (Go/Redis)     │  │
│  └─────────────┘  └─────────────┘  └───────────────────┘  │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │ Agent Pool  │  │ Monitoring  │  │   Model Server    │  │
│  │  (Python)   │  │ (Grafana)   │  │    (Python)       │  │
│  └─────────────┘  └─────────────┘  └───────────────────┘  │
│                                                             │
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

### 3. Experience Buffer Service

**File**: `docker/experience-buffer/Dockerfile`

```dockerfile
FROM golang:1.24-alpine AS builder

WORKDIR /build

# Copy and download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build experience service
RUN CGO_ENABLED=0 go build \
    -o experience-service \
    ./cmd/experience_service

# Runtime
FROM alpine:3.21

RUN apk add --no-cache ca-certificates

WORKDIR /app

COPY --from=builder /build/experience-service .

EXPOSE 50052

HEALTHCHECK CMD ["/app/experience-service", "health"] || exit 1

ENTRYPOINT ["/app/experience-service"]
```

### 4. Development Environment

**File**: `docker/dev/Dockerfile`

```dockerfile
# Full development environment with all tools
FROM golang:1.24

# Install additional tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    protobuf-compiler \
    make \
    git \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Go tools
RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
RUN go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Install Python tools
RUN pip install --no-cache-dir \
    black \
    flake8 \
    mypy \
    pytest \
    ipython

WORKDIR /workspace

# Keep container running
CMD ["sleep", "infinity"]
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

## Build and CI/CD Recommendations

### 1. Multi-Architecture Builds

```bash
# Build script for multiple architectures
#!/bin/bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag generals-game-server:latest \
  --push \
  -f docker/game-server/Dockerfile .
```

### 2. GitHub Actions Workflow

**File**: `.github/workflows/docker-build.yml`

```yaml
name: Docker Build and Push

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push Game Server
        uses: docker/build-push-action@v4
        with:
          context: .
          file: docker/game-server/Dockerfile
          push: true
          tags: |
            ${{ steps.login-ecr.outputs.registry }}/generals-game-server:latest
            ${{ steps.login-ecr.outputs.registry }}/generals-game-server:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

## Development Workflow

### 1. Local Development with Hot Reload

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

### 2. Running RL Training Locally

```bash
# Start training environment
docker-compose -f docker-compose.dev.yml up -d game-server experience-buffer redis

# Scale up agents for self-play
docker-compose -f docker-compose.dev.yml up -d --scale rl-agent=8

# Monitor training
docker-compose -f docker-compose.dev.yml logs -f rl-agent
```

## Best Practices and Guidelines

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

## Cloud Deployment Considerations

### AWS ECS Task Definitions

```json
{
  "family": "generals-game-server",
  "taskRoleArn": "arn:aws:iam::account:role/generals-task-role",
  "executionRoleArn": "arn:aws:iam::account:role/generals-execution-role",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "game-server",
      "image": "${ECR_REGISTRY}/generals-game-server:latest",
      "cpu": 2048,
      "memory": 4096,
      "portMappings": [
        {
          "containerPort": 50051,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/generals",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "game-server"
        }
      }
    }
  ]
}
```

### Kubernetes Manifests

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: game-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: game-server
  template:
    metadata:
      labels:
        app: game-server
    spec:
      containers:
      - name: game-server
        image: generals-game-server:latest
        ports:
        - containerPort: 50051
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          exec:
            command: ["/app/game-server", "health"]
          initialDelaySeconds: 10
          periodSeconds: 30
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

### Phase 4: Production Optimization (Month 2)
1. Optimize image sizes
2. Implement caching strategies
3. Set up A/B testing for models
4. Add distributed training support

## Additional Files Needed

### 1. `.dockerignore`
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

### 2. `docker/scripts/entrypoint.sh`
```bash
#!/bin/sh
set -e

# Wait for dependencies
if [ -n "$WAIT_FOR_SERVICES" ]; then
    for service in $WAIT_FOR_SERVICES; do
        echo "Waiting for $service..."
        while ! nc -z $service 2>/dev/null; do
            sleep 1
        done
        echo "$service is ready"
    done
fi

# Run migrations or setup if needed
if [ "$RUN_SETUP" = "true" ]; then
    echo "Running setup..."
    # Add setup commands here
fi

# Execute main command
exec "$@"
```

### 3. `Makefile` additions
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

## Monitoring and Observability

### 1. Structured Logging
- Use JSON logging format
- Include trace IDs for request correlation
- Log to stdout for container compatibility

### 2. Metrics Collection
- Expose Prometheus metrics endpoints
- Track key performance indicators
- Monitor resource usage

### 3. Distributed Tracing
- Implement OpenTelemetry
- Track request flow across services
- Identify performance bottlenecks

## Conclusion

This Docker strategy provides a solid foundation for both local development and cloud deployment of your RL training infrastructure. The modular approach allows for independent scaling of components and supports the distributed nature of RL training workloads.

Key benefits:
- **Modularity**: Each service in its own container
- **Scalability**: Easy to scale individual components
- **Portability**: Consistent environments across dev/prod
- **Efficiency**: Optimized images and resource usage
- **Observability**: Built-in monitoring and logging

Next steps:
1. Create the Docker files as specified
2. Test locally with docker-compose
3. Set up CI/CD pipelines
4. Deploy to cloud infrastructure