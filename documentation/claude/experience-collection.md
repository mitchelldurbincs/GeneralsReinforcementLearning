# Experience Collection Implementation Guide

## Overview

This document provides a detailed implementation plan for the experience collection system in the Generals.io RL training infrastructure. The system collects (state, action, reward, next_state, done) tuples during gameplay for training reinforcement learning agents.

## Current Implementation Status

### ✅ Completed Components

1. **Experience Protobuf Messages** (`proto/experience/v1/experience.proto`)
   - Defines `Experience` message with tensor states, actions, rewards
   - Includes `TensorState` for multi-channel neural network inputs
   - Supports experience streaming and batch submission

2. **State Serializer** (`internal/experience/serializer.go`)
   - Converts game states to 9-channel tensor representation:
     - Channel 0: Own armies (normalized 0-1)
     - Channel 1: Enemy armies (normalized 0-1)
     - Channel 2: Own territory (binary)
     - Channel 3: Enemy territory (binary)
     - Channel 4: Neutral territory (binary)
     - Channel 5: Cities (binary)
     - Channel 6: Mountains (binary)
     - Channel 7: Visible tiles (binary)
     - Channel 8: Fog of war tiles (binary)
   - Provides action mask generation for legal moves
   - Handles action index conversion (flat index ↔ x,y,direction)

3. **Reward Calculator** (`internal/experience/rewards.go`)
   - Implements reward structure:
     - Win/Loss: +1.0/-1.0
     - City capture/loss: +0.1/-0.1
     - Army advantage: Proportional scaling from -0.05 to +0.05
   - Configurable reward weights via `RewardConfig`

4. **Experience Buffer** (`internal/experience/buffer.go`)
   - Thread-safe circular buffer for experiences
   - Supports streaming via channels
   - Provides batch sampling for training
   - Includes save/load functionality

5. **Game Engine Integration** (`internal/game/engine.go`)
   - Experience collector interface to avoid circular dependencies
   - Hooks in `Step()` function to capture state transitions
   - Automatic experience collection when games end

## Remaining Implementation Tasks

### 1. gRPC StreamExperiences Endpoint

**Purpose**: Allow trainers to consume experiences in real-time from the game server.

**Implementation Plan**:

```go
// internal/grpc/gameserver/experience_service.go
package gameserver

import (
    "context"
    experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
)

type ExperienceService struct {
    experiencepb.UnimplementedExperienceServiceServer
    bufferManager *experience.BufferManager
}

func (s *ExperienceService) StreamExperiences(
    req *experiencepb.StreamExperiencesRequest,
    stream experiencepb.ExperienceService_StreamExperiencesServer,
) error {
    // 1. Validate request parameters
    // 2. Create filtered stream based on game_ids, player_ids
    // 3. Stream experiences as they arrive
    // 4. Handle batching if requested
    // 5. Implement follow mode for continuous streaming
}
```

**Key Features**:
- Filter by game IDs and player IDs
- Batch experiences for efficient network usage
- Follow mode to continuously stream new experiences
- Backpressure handling to prevent overwhelming trainers

### 2. Experience Writer for Persistent Storage

**Purpose**: Save experiences to disk or S3 for replay training and fault tolerance.

**Implementation Plan**:

```go
// internal/experience/writer.go
package experience

type Writer interface {
    Write(experiences []*experiencepb.Experience) error
    Close() error
}

// File-based writer using TFRecord format
type TFRecordWriter struct {
    file       *os.File
    compressor *gzip.Writer
    index      *IndexFile
}

// S3-based writer with partitioning
type S3Writer struct {
    client     *s3.Client
    bucket     string
    prefix     string
    partitioner *TimePartitioner
}

// Composite writer for multiple destinations
type MultiWriter struct {
    writers []Writer
}
```

**Storage Formats**:
- **TFRecord**: Compatible with TensorFlow training pipelines
- **NPZ**: NumPy compressed format for PyTorch
- **Parquet**: Columnar format for analytics and batch processing

**Partitioning Strategy**:
```
s3://bucket/experiences/
  └── year=2024/
      └── month=01/
          └── day=15/
              └── hour=14/
                  ├── game_abc123_0000.tfrecord
                  ├── game_abc123_0001.tfrecord
                  └── _index.json
```

### 3. Python Trainer Integration

**Purpose**: Enable Python-based RL trainers to consume experiences from the Go server.

**Implementation Plan**:

```python
# python/generals_agent/experience_consumer.py
import grpc
from typing import Iterator, Optional, List
import numpy as np
from generals_pb.experience.v1 import experience_pb2, experience_pb2_grpc

class ExperienceConsumer:
    """Consumes experiences from the Go game server for training."""
    
    def __init__(self, server_address: str):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = experience_pb2_grpc.ExperienceServiceStub(self.channel)
    
    def stream_experiences(
        self,
        game_ids: Optional[List[str]] = None,
        player_ids: Optional[List[int]] = None,
        batch_size: int = 32,
        follow: bool = True
    ) -> Iterator[List[Experience]]:
        """Stream experiences from the server."""
        request = experience_pb2.StreamExperiencesRequest(
            game_ids=game_ids or [],
            player_ids=player_ids or [],
            batch_size=batch_size,
            follow=follow
        )
        
        batch = []
        for exp in self.stub.StreamExperiences(request):
            # Convert protobuf to numpy arrays
            experience = self._proto_to_numpy(exp)
            batch.append(experience)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
    
    def _proto_to_numpy(self, exp: experience_pb2.Experience):
        """Convert protobuf experience to numpy arrays."""
        state = np.array(exp.state.data).reshape(exp.state.shape)
        next_state = np.array(exp.next_state.data).reshape(exp.next_state.shape)
        action_mask = np.array(exp.action_mask)
        
        return Experience(
            state=state,
            action=exp.action,
            reward=exp.reward,
            next_state=next_state,
            done=exp.done,
            action_mask=action_mask
        )
```

**Training Loop Integration**:
```python
# python/generals_agent/trainer.py
class DistributedTrainer:
    def __init__(self, experience_server: str, model: nn.Module):
        self.consumer = ExperienceConsumer(experience_server)
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters())
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
    
    def train(self):
        # Start experience streaming in background
        experience_thread = threading.Thread(
            target=self._consume_experiences,
            daemon=True
        )
        experience_thread.start()
        
        while True:
            # Sample batch from replay buffer
            batch = self.replay_buffer.sample(batch_size=256)
            
            # Training step
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

### 4. Configuration Management

**Purpose**: Make experience collection configurable without code changes.

**Implementation Plan**:

```yaml
# config/experience.yaml
experience:
  collection:
    enabled: true
    buffer_size: 10000
    
  streaming:
    max_connections: 100
    batch_timeout_ms: 100
    
  storage:
    enabled: true
    type: "s3"  # or "file"
    s3:
      bucket: "generals-experiences"
      prefix: "training/"
      region: "us-east-1"
    file:
      directory: "/data/experiences"
      format: "tfrecord"  # or "npz"
      
  rewards:
    win_game: 1.0
    lose_game: -1.0
    city_captured: 0.1
    city_lost: -0.1
```

**Dynamic Configuration Updates**:
```go
// internal/experience/config.go
type ConfigManager struct {
    viper *viper.Viper
    mu    sync.RWMutex
    
    onUpdate []func(*Config)
}

func (cm *ConfigManager) Watch() {
    viper.WatchConfig()
    viper.OnConfigChange(func(e fsnotify.Event) {
        cm.mu.Lock()
        defer cm.mu.Unlock()
        
        newConfig := cm.loadConfig()
        for _, callback := range cm.onUpdate {
            callback(newConfig)
        }
    })
}
```

### 5. Monitoring and Metrics

**Purpose**: Track experience collection performance and health.

**Metrics to Track**:
- Experiences collected per second
- Buffer utilization
- Storage write latency
- Stream consumer lag
- Reward statistics (mean, std, min, max)

**Implementation**:
```go
// internal/experience/metrics.go
var (
    experiencesCollected = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "experiences_collected_total",
            Help: "Total number of experiences collected",
        },
        []string{"game_id", "player_id"},
    )
    
    bufferSize = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "experience_buffer_size",
            Help: "Current size of experience buffer",
        },
        []string{"buffer_id"},
    )
    
    rewardHistogram = prometheus.NewHistogram(
        prometheus.HistogramOpts{
            Name: "experience_rewards",
            Help: "Distribution of experience rewards",
            Buckets: prometheus.LinearBuckets(-1, 0.1, 21),
        },
    )
)
```

## Testing Strategy

### Unit Tests

```go
// internal/experience/serializer_test.go
func TestStateToTensor(t *testing.T) {
    // Test state serialization produces correct tensor shape
    // Test normalization of army values
    // Test visibility encoding
}

// internal/experience/rewards_test.go
func TestRewardCalculation(t *testing.T) {
    // Test win/loss rewards
    // Test city capture rewards
    // Test army advantage calculation
}
```

### Integration Tests

```go
// internal/experience/integration_test.go
func TestEndToEndExperienceCollection(t *testing.T) {
    // Create game with experience collection
    // Play several turns
    // Verify experiences are collected correctly
    // Check reward calculations
}
```

### Load Tests

```python
# tests/load_test_experience_streaming.py
def test_concurrent_consumers():
    """Test multiple trainers consuming experiences concurrently."""
    consumers = [ExperienceConsumer(SERVER) for _ in range(10)]
    
    # Start consuming in parallel
    futures = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for consumer in consumers:
            future = executor.submit(
                consume_n_experiences, 
                consumer, 
                n=10000
            )
            futures.append(future)
    
    # Verify all consumers received experiences
    for future in futures:
        assert future.result() == 10000
```

## Deployment Considerations

### 1. Resource Requirements

- **Memory**: ~10GB for 1M experiences in buffer
- **Storage**: ~1GB/hour at 1000 games/second
- **Network**: ~100Mbps for 10 concurrent trainers

### 2. Scaling Strategy

```yaml
# kubernetes/experience-service.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: experience-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: game-server
  minReplicas: 3
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: experience_buffer_utilization
      target:
        type: AverageValue
        averageValue: "80"
```

### 3. Fault Tolerance

- **Persistent buffers**: Write experiences to disk before acknowledgment
- **Replay on failure**: Trainers can resume from last checkpoint
- **Redundant storage**: Write to multiple destinations
- **Health checks**: Automatic detection and recovery of failed components

## Performance Optimizations

### 1. Zero-Copy Serialization

```go
// Use memory pooling for tensor allocations
var tensorPool = sync.Pool{
    New: func() interface{} {
        return make([]float32, 9*20*20) // Pre-allocate for 20x20 board
    },
}
```

### 2. Batch Processing

```go
// Process multiple experiences in a single operation
func (c *Collector) OnBatchStateTransition(
    transitions []StateTransition,
) []*experiencepb.Experience {
    experiences := make([]*experiencepb.Experience, 0, len(transitions))
    
    // Process in parallel
    var wg sync.WaitGroup
    expChan := make(chan *experiencepb.Experience, len(transitions))
    
    for _, t := range transitions {
        wg.Add(1)
        go func(trans StateTransition) {
            defer wg.Done()
            exp := c.processTransition(trans)
            expChan <- exp
        }(t)
    }
    
    wg.Wait()
    close(expChan)
    
    for exp := range expChan {
        experiences = append(experiences, exp)
    }
    
    return experiences
}
```

### 3. Compression

```go
// Compress experiences before storage/transmission
func CompressExperiences(experiences []*experiencepb.Experience) ([]byte, error) {
    var buf bytes.Buffer
    
    // Use zstd for better compression ratio
    w, _ := zstd.NewWriter(&buf, zstd.WithEncoderLevel(zstd.SpeedFastest))
    
    for _, exp := range experiences {
        data, _ := proto.Marshal(exp)
        binary.Write(w, binary.LittleEndian, uint32(len(data)))
        w.Write(data)
    }
    
    w.Close()
    return buf.Bytes(), nil
}
```

## Future Enhancements

1. **Prioritized Experience Replay**: Weight experiences by TD-error
2. **Distributed Experience Buffer**: Redis/Hazelcast for shared buffer
3. **Experience Augmentation**: Generate synthetic experiences
4. **Curriculum Learning**: Adjust game difficulty based on agent performance
5. **Multi-Agent Experience Sharing**: Share experiences between similar agents