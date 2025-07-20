# Deployment Architecture

## Overview

This document outlines a simplified AWS deployment architecture for the Generals.io Reinforcement Learning game, designed for personal use with occasional gameplay against the trained bot.

## Architecture Design

### Minimal Architecture for Personal Use

Since this is for personal use with occasional gameplay, we'll use a cost-effective, simple architecture:

```
Internet → EC2 Instance (Game Server + RL Bot)
              ↓
         gRPC Server
              ↓
      Local UI Client
```

## AWS Components

### 1. **Single EC2 Instance**
- **Instance Type**: `t3.medium` for development/training, `t3.micro` for just playing
- **Purpose**: Runs both game server and trained RL model
- **OS**: Amazon Linux 2023 or Ubuntu 22.04
- **Storage**: 30GB EBS volume for models and game data

### 2. **Container Registry (ECR)**
- Store Docker images for the game server
- Versioned deployments
- Private repository for security

### 3. **S3 Bucket**
- Store trained RL models
- Game replay storage
- Training checkpoints
- Cost-effective long-term storage

### 4. **Networking**
- **VPC**: Default VPC is sufficient
- **Security Group**: 
  - SSH (port 22) from your IP
  - gRPC (port 50051) from your IP
  - HTTP/HTTPS if adding web interface later
- **Elastic IP**: Optional for consistent connection

### 5. **Simple Monitoring**
- CloudWatch for basic metrics
- CloudWatch Logs for application logs
- SNS alerts for instance health

## Deployment Strategy

### Phase 1: Development Setup
```bash
# Local development
docker build -t generals-game .
docker run -p 50051:50051 generals-game

# Connect with UI client locally
go run cmd/ui_client/main.go
```

### Phase 2: AWS Deployment
1. Build and push Docker image to ECR
2. Launch EC2 instance with Docker installed
3. Pull and run container from ECR
4. Connect UI client to EC2's public IP

### Phase 3: RL Training
1. Use same EC2 instance for training (upgrade to GPU instance if needed)
2. Store checkpoints in S3
3. Load best model for gameplay

## Cost Optimization

### Estimated Monthly Costs (Personal Use)
- **EC2 t3.micro**: ~$8/month (always on)
- **EBS Storage**: ~$3/month (30GB)
- **S3**: <$1/month (model storage)
- **ECR**: <$1/month (minimal images)
- **Total**: ~$13/month

### Cost Saving Tips
1. **Stop instance when not using** - reduces to ~$3/month
2. **Use Spot instances for training** - 70% cost reduction
3. **Schedule automatic start/stop** - play on weekends only
4. **Local development first** - minimize cloud usage

## Simplified Terraform Structure

```hcl
# main.tf
resource "aws_instance" "game_server" {
  ami           = "ami-0c02fb55956c7d316"  # Amazon Linux 2023
  instance_type = "t3.micro"
  
  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install -y docker
    service docker start
    # Pull and run container from ECR
  EOF
  
  tags = {
    Name = "generals-game-server"
  }
}

resource "aws_s3_bucket" "game_data" {
  bucket = "generals-game-data"
  
  lifecycle_rule {
    enabled = true
    
    transition {
      days          = 30
      storage_class = "INFREQUENT_ACCESS"
    }
  }
}

resource "aws_ecr_repository" "game_server" {
  name = "generals-game-server"
}
```

## Quick Start Commands

### 1. Build and Deploy
```bash
# Build Docker image
docker build -f deploy/Dockerfile -t generals-game .

# Tag for ECR
docker tag generals-game:latest [aws-account].dkr.ecr.us-east-1.amazonaws.com/generals-game:latest

# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin [aws-account].dkr.ecr.us-east-1.amazonaws.com
docker push [aws-account].dkr.ecr.us-east-1.amazonaws.com/generals-game:latest
```

### 2. Run on EC2
```bash
# SSH to instance
ssh ec2-user@[instance-ip]

# Pull and run
docker pull [aws-account].dkr.ecr.us-east-1.amazonaws.com/generals-game:latest
docker run -d -p 50051:50051 generals-game:latest
```

### 3. Connect Client
```bash
# Set server address in client config or environment
export GAME_SERVER_ADDRESS="[instance-ip]:50051"
go run cmd/ui_client/main.go
```

## Future Enhancements

If you want to expand later:
1. **Web UI**: Add web-based client with WebSocket support
2. **Multiple Bots**: Train different difficulty levels
3. **Friends Access**: Add authentication and open to friends
4. **Tournament Mode**: Schedule bot vs bot matches
5. **Analytics**: Track game statistics and bot performance

## Security Notes

For personal use:
- Restrict Security Group to your home IP
- Use SSH keys, not passwords
- Keep ECR repository private
- Enable CloudTrail for audit logs
- Regular security updates on EC2

## Maintenance

Monthly tasks:
1. Update EC2 instance OS
2. Review CloudWatch logs
3. Backup trained models to S3
4. Clean up old Docker images in ECR
5. Check AWS bill for anomalies