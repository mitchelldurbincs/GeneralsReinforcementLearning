# Reinforcement Learning Cloud Architecture

## Overview

Based on your services.md and existing infrastructure, here's how the RL components will work in the cloud:

## Service Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Public Internet                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────┴──────────┐
                          │   Load Balancer    │
                          │  (ALB/NLB for gRPC)│
                          └─────────┬──────────┘
                                    │
┌───────────────────────────────────┴─────────────────────────────────┐
│                         VPC (10.0.0.0/16)                           │
│                                                                      │
│  ┌─────────────────────── Public Subnet ──────────────────────┐    │
│  │                                                              │    │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │    │
│  │  │  Game Service   │  │ Matchmaker Svc   │  │  API GW    │ │    │
│  │  │  (EC2/ECS)      │  │   (EC2/ECS)      │  │            │ │    │
│  │  │  Port: 50051    │  │   Port: 50052    │  │            │ │    │
│  │  └────────┬────────┘  └────────┬─────────┘  └──────┬─────┘ │    │
│  │           │                    │                     │       │    │
│  └───────────┼────────────────────┼─────────────────────┼──────┘    │
│              │                    │                     │            │
│  ┌───────────┴────────────────────┴─────────────────────┴──────┐    │
│  │                    Private Subnet A                          │    │
│  │                                                              │    │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │    │
│  │  │  Model Service  │  │   Elo Service    │  │  Replay    │ │    │
│  │  │  (EC2/ECS)      │  │   (EC2/ECS)      │  │  Service   │ │    │
│  │  │  Port: 50053    │  │   Port: 50054    │  │ Port:50055 │ │    │
│  │  └────────┬────────┘  └────────┬─────────┘  └──────┬─────┘ │    │
│  │           │                    │                     │       │    │
│  └───────────┼────────────────────┼─────────────────────┼──────┘    │
│              │                    │                     │            │
│  ┌───────────┴────────────────────┴─────────────────────┴──────┐    │
│  │                    Private Subnet B (GPU)                    │    │
│  │                                                              │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │            RL Training Cluster (GPU Instances)       │    │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │    │
│  │  │  │  Learner 1  │  │  Learner 2  │  │  Learner N  │ │    │    │
│  │  │  │ (p3.2xlarge)│  │ (p3.2xlarge)│  │ (p3.2xlarge)│ │    │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │    │
│  │  │  │   Actor 1   │  │   Actor 2   │  │   Actor N   │ │    │    │
│  │  │  │ (c5.xlarge) │  │ (c5.xlarge) │  │ (c5.xlarge) │ │    │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

## Data Storage Layer

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   S3 Buckets    │  │    RDS/Aurora   │  │   ElastiCache   │
│                 │  │   (PostgreSQL)   │  │     (Redis)     │
│ • Model Weights │  │                 │  │                 │
│ • Replays       │  │ • Game Metadata │  │ • Elo Rankings  │
│ • Training Data │  │ • Player Stats  │  │ • Active Games  │
│ • Checkpoints   │  │ • Match History │  │ • Session Data  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Service Interactions

### 1. Model Service (Port 50053)
**Location**: Private Subnet A  
**Instance Type**: c5.2xlarge or m5.2xlarge  
**Purpose**: Centralized model management and distribution

**Key Functions**:
- Stores and versions trained models in S3
- Serves model weights to actors via GetPolicy RPC
- Validates new models before deployment
- Manages model lifecycle (promotion, rollback)

**Interactions**:
- **From Learners**: Receive PublishPolicy calls with new model weights
- **To Actors**: Serve GetPolicy requests for latest models
- **To S3**: Store/retrieve model artifacts
- **To Monitoring**: Report model performance metrics

### 2. Elo Service (Port 50054)
**Location**: Private Subnet A  
**Instance Type**: t3.medium (can be lightweight)  
**Purpose**: Track and update agent performance ratings

**Key Functions**:
- Maintain Elo ratings for different model versions
- Update ratings based on match outcomes
- Provide rankings for matchmaking

**Interactions**:
- **From Game Service**: Receive match results
- **To Database**: Store/update Elo ratings
- **To Matchmaker**: Provide ratings for pairing decisions
- **To Redis**: Cache frequently accessed rankings

### 3. Matchmaker Service (Port 50052)
**Location**: Public Subnet (needs external access)  
**Instance Type**: c5.large  
**Purpose**: Pair agents/players for games

**Key Functions**:
- Match players/agents based on Elo ratings
- Handle tournament bracket generation
- Manage queue for waiting players/agents

**Interactions**:
- **From Clients/Actors**: Receive FindMatch requests
- **To Elo Service**: Query ratings for fair matchmaking
- **To Game Service**: Create new game instances
- **To Redis**: Manage matchmaking queues

### 4. Replay Service (Port 50055)
**Location**: Private Subnet A  
**Instance Type**: r5.xlarge (memory-optimized for buffering)  
**Purpose**: Collect and distribute training experiences

**Key Functions**:
- Buffer experiences from multiple actors
- Implement prioritized experience replay
- Serve batches to learners

**Interactions**:
- **From Actors**: Receive RecordExperience calls
- **To Learners**: Serve GetExperienceBatch requests
- **To S3**: Archive old experiences
- **To Redis**: Maintain priority queue for PER

## RL Training Flow

### 1. Self-Play Loop
```
Actor → Game Service → Generate Experience → Replay Service
  ↓                                              ↓
Model Service ← Learner ← Experience Batch ←────┘
```

### 2. Evaluation Loop
```
Model Service → Matchmaker → Game Service → Elo Service
      ↓             ↓                           ↓
   New Model    Tournament                 Update Ratings
```

## AWS Service Mapping

### Compute Resources
- **Game/Matchmaker Services**: ECS Fargate or EC2 with Auto Scaling
- **Model/Elo/Replay Services**: ECS or EC2 in private subnets
- **RL Training Cluster**:
  - Learners: p3.2xlarge (GPU) or p3.8xlarge for faster training
  - Actors: c5.xlarge or c5.2xlarge (CPU-optimized)

### Storage
- **S3 Buckets**:
  - `generals-rl-models/`: Trained model storage
  - `generals-rl-replays/`: Game replay archive
  - `generals-rl-experiences/`: Training data buffer
- **RDS Aurora PostgreSQL**: Game metadata, player stats
- **ElastiCache Redis**: Fast access to active data

### Networking
- **VPC**: Custom VPC with public/private subnets
- **Security Groups**:
  - Public services: Allow gRPC ports from internet
  - Private services: Only VPC internal traffic
  - Training cluster: Isolated with specific port access
- **VPC Endpoints**: Direct S3 access from private subnets

### Scaling Strategy
- **Horizontal Scaling**:
  - Actors: Scale based on game queue length
  - Learners: Scale based on experience buffer size
- **Vertical Scaling**:
  - Upgrade GPU instances for faster training
  - Use Spot instances for cost-effective actor pool

## Cost Optimization

### Production Setup (~$500-1000/month)
- 2-3 GPU instances for learners (Spot)
- 5-10 CPU instances for actors (Spot)
- Small instances for services
- Storage and data transfer costs

### Development Setup (~$50-100/month)
- 1 GPU instance (on-demand, stopped when not training)
- 2-3 CPU instances for actors
- Minimal service instances
- Aggressive S3 lifecycle policies

### Personal/Hobby Setup (~$20-30/month)
- No dedicated GPU (train locally, deploy models)
- Single EC2 instance running all services
- Minimal storage
- Stop instances when not in use

## Deployment Recommendations

1. **Start Small**: Begin with all services on one EC2 instance
2. **Containerize**: Use Docker for all services
3. **Use Terraform**: Define infrastructure as code
4. **Monitor Costs**: Set up billing alerts
5. **Implement Auto-shutdown**: Stop training resources when idle
6. **Use Spot Instances**: For training cluster (70% cost savings)
7. **Cache Aggressively**: Reduce inter-service calls
8. **Batch Operations**: Group experience uploads/downloads