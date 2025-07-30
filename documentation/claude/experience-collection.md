# Experience Collection Implementation Guide

## Overview

This document provides a detailed implementation plan for the experience collection system in the Generals.io RL training infrastructure. The system collects (state, action, reward, next_state, done) tuples during gameplay for training reinforcement learning agents.

## Architecture Overview

### Data Flow Diagram
```
Game Engine → Experience Collector → Experience Buffer → Storage/Streaming
     ↓              ↓                      ↓                    ↓
Game State    State Tensor          In-Memory Queue      S3/File Storage
             Reward Calc                   ↓              gRPC Stream
             Action Mask             Python Trainer ←─────────┘
```

### Processing Paths
1. **Online Path** (Real-time streaming): Game → Collector → Buffer → gRPC Stream → Trainer
2. **Offline Path** (Batch processing): Game → Collector → Buffer → Storage → Batch Trainer

## Current Implementation Status

### ✅ Actually Completed Components

1. **Experience Protobuf Messages** (`proto/experience/v1/experience.proto`)
   - Defines `Experience` message with tensor states, actions, rewards
   - Includes `TensorState` for multi-channel neural network inputs
   - Supports experience streaming and batch submission

2. **Game Engine Integration** (`internal/game/engine.go`)
   - Experience collector interface (`internal/game/experience_collector.go`)
   - Hooks in `Step()` function to capture state transitions
   - Support for experience collection when games end

3. **Core Experience Collection Components** (Added January 2025)
   - **SimpleCollector** (`internal/experience/collector.go`) - Basic in-memory experience collector
   - **Serializer** (`internal/experience/serializer.go`) - State to tensor conversion
   - **Reward Calculator** (`internal/experience/rewards.go`) - Configurable reward computation
   - **Experience Buffer** (`internal/experience/buffer.go`) - Thread-safe circular buffer with streaming

### 🐛 Issues Fixed (January 2025)

1. **Test Compilation Errors**:
   - Fixed incorrect tile type constants (e.g., `core.TileTypeEmpty` → `core.TileNormal`)
   - Fixed tile struct field names (e.g., `Tiles` → `T`, `Armies` → `Army`)
   - Fixed visibility method calls (e.g., `SetVisibleTo()` → `SetVisible()`)
   - Removed unused imports

2. **Game State API Mismatches**:
   - GameState doesn't have `GameOver` or `Winner` fields directly
   - Must use `IsGameOver()` and `GetWinner()` methods instead
   - These require Engine integration, so some tests are skipped

### ⚠️ Partially Implemented / Needs Work

1. **Test Coverage**:
   - Most tests pass except `TestSimpleCollector_OnStateTransition` which has an assertion failure
   - Win/loss reward tests are skipped as they require Engine integration
   - Need to add integration tests with actual game engine

2. **Missing gRPC Integration**:
   - StreamExperiences endpoint not yet implemented
   - Need to connect buffer to gRPC streaming service

## MVP Implementation Strategy

### Phase 1: Core Components (Week 1)
Start with a minimal viable implementation to prove the concept:

```go
// internal/experience/collector_simple.go
package experience

import (
    "sync"
    "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
    experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
)

type SimpleCollector struct {
    experiences []*experiencepb.Experience
    mu          sync.Mutex
    maxSize     int
}

func NewSimpleCollector(maxSize int) *SimpleCollector {
    return &SimpleCollector{
        experiences: make([]*experiencepb.Experience, 0, maxSize),
        maxSize:     maxSize,
    }
}

func (c *SimpleCollector) OnStateTransition(prevState, currState *game.GameState, actions map[int]*game.Action) {
    // TODO: Implement state serialization
    // TODO: Calculate rewards
    // TODO: Create experience protobuf
}

func (c *SimpleCollector) OnGameEnd(finalState *game.GameState) {
    // TODO: Handle terminal states
}

func (c *SimpleCollector) GetExperiences() []*experiencepb.Experience {
    c.mu.Lock()
    defer c.mu.Unlock()
    return append([]*experiencepb.Experience{}, c.experiences...)
}
```

### Phase 2: Serialization (Week 1-2)
Implement the state tensor conversion:

```go
// internal/experience/serializer.go
package experience

import (
    "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
)

const (
    ChannelOwnArmies = 0
    ChannelEnemyArmies = 1
    ChannelOwnTerritory = 2
    ChannelEnemyTerritory = 3
    ChannelNeutralTerritory = 4
    ChannelCities = 5
    ChannelMountains = 6
    ChannelVisible = 7
    ChannelFog = 8
    NumChannels = 9
)

func StateToTensor(state *game.GameState, playerID int) []float32 {
    width := state.Board.Width
    height := state.Board.Height
    tensor := make([]float32, NumChannels*width*height)
    
    // TODO: Implement channel filling logic
    
    return tensor
}
```

### Phase 3: Basic Rewards (Week 2)
Simple reward calculation without configuration:

```go
// internal/experience/rewards.go
package experience

func CalculateReward(prevState, currState *game.GameState, playerID int) float32 {
    // Start simple - just win/loss
    if currState.Winner == playerID {
        return 1.0
    } else if currState.Winner != -1 {
        return -1.0
    }
    return 0.0 // Game continues
}
```

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

### Performance Benchmarks

```go
// internal/experience/bench_test.go
func BenchmarkStateToTensor(b *testing.B) {
    state := createTestGameState(20, 20) // 20x20 board
    serializer := NewSerializer()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _ = serializer.StateToTensor(state, 0)
    }
    b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "states/sec")
}

func BenchmarkExperienceCompression(b *testing.B) {
    experiences := generateTestExperiences(100)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, _ = CompressExperiences(experiences)
    }
    
    // Report compression ratio
    original := calculateSize(experiences)
    compressed, _ := CompressExperiences(experiences)
    b.ReportMetric(float64(original)/float64(len(compressed)), "compression_ratio")
}

func BenchmarkMemoryPooling(b *testing.B) {
    b.Run("WithPooling", func(b *testing.B) {
        serializer := NewPooledSerializer()
        state := createTestGameState(20, 20)
        
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            buf := serializer.StateToTensorPooled(state, 0)
            buf.Release()
        }
    })
    
    b.Run("WithoutPooling", func(b *testing.B) {
        serializer := NewSerializer()
        state := createTestGameState(20, 20)
        
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            _ = serializer.StateToTensor(state, 0)
        }
    })
}
```

### Memory Profiling Tests

```go
// internal/experience/memory_test.go
func TestBufferMemoryUsage(t *testing.T) {
    var m runtime.MemStats
    
    runtime.GC()
    runtime.ReadMemStats(&m)
    allocBefore := m.Alloc
    
    buffer := NewExperienceBuffer(100000)
    for i := 0; i < 100000; i++ {
        buffer.Add(generateTestExperience())
    }
    
    runtime.GC()
    runtime.ReadMemStats(&m)
    allocAfter := m.Alloc
    
    bytesPerExperience := float64(allocAfter-allocBefore) / 100000
    t.Logf("Memory per experience: %.2f KB", bytesPerExperience/1024)
    
    // Ensure we're within expected bounds
    assert.Less(t, bytesPerExperience, 35000.0) // Less than 35KB per experience
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

def test_experience_throughput():
    """Measure maximum experience collection throughput."""
    collector = ExperienceCollector()
    
    start_time = time.time()
    experiences_collected = 0
    
    # Run for 60 seconds
    while time.time() - start_time < 60:
        state = generate_random_state()
        next_state = generate_random_state()
        
        collector.collect(state, action=0, reward=0.0, next_state=next_state, done=False)
        experiences_collected += 1
    
    throughput = experiences_collected / 60
    print(f"Throughput: {throughput:.2f} experiences/second")
    
    # Should handle at least 50k experiences/second
    assert throughput > 50000
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

## Performance Considerations and Optimizations

### Memory and Bandwidth Analysis

For a 20x20 board:
- State size: 9 channels × 20 × 20 × 4 bytes = 14.4KB per state
- Experience size: ~30KB (state + next_state + metadata)
- At 1000 games/second with 50 turns average:
  - Raw data rate: 1.5GB/s
  - After compression (zstd): ~300-500MB/s
  - Network bandwidth: ~2.4-4Gbps

### Memory Requirements by Scale
| Games/Second | Buffer Size (1M exp) | Data Rate | Network BW |
|--------------|---------------------|-----------|------------|
| 100          | 3GB                 | 150MB/s   | 1.2Gbps    |
| 500          | 15GB                | 750MB/s   | 6Gbps      |
| 1000         | 30GB                | 1.5GB/s   | 12Gbps     |

### 1. Zero-Copy Serialization

```go
// Use memory pooling for tensor allocations
var tensorPool = sync.Pool{
    New: func() interface{} {
        return &TensorBuffer{
            data: make([]float32, 9*20*20), // Pre-allocate for 20x20 board
        }
    },
}

type TensorBuffer struct {
    data []float32
    size int
}

func (s *Serializer) StateToTensorPooled(state *game.GameState, playerID int) *TensorBuffer {
    buf := tensorPool.Get().(*TensorBuffer)
    buf.size = NumChannels * state.Board.Width * state.Board.Height
    
    // Reuse buffer if it's large enough
    if len(buf.data) < buf.size {
        buf.data = make([]float32, buf.size)
    }
    
    // Fill tensor data...
    return buf
}

// Return buffer to pool when done
func (buf *TensorBuffer) Release() {
    tensorPool.Put(buf)
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

### 3. Compression Strategy

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

### 4. Early Compression in Pipeline

```go
// Compress at collection time to reduce memory usage
type CompressedCollector struct {
    compressed [][]byte
    compressor *zstd.Encoder
    mu         sync.Mutex
}

func (c *CompressedCollector) AddExperience(exp *experiencepb.Experience) error {
    data, err := proto.Marshal(exp)
    if err != nil {
        return err
    }
    
    c.mu.Lock()
    defer c.mu.Unlock()
    
    compressed := c.compressor.EncodeAll(data, nil)
    c.compressed = append(c.compressed, compressed)
    
    return nil
}
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. High Memory Usage
**Symptoms**: OOM errors, slow performance, system instability

**Solutions**:
- Reduce buffer size in configuration
- Enable compression earlier in pipeline
- Implement experience sampling/dropping
- Use disk-backed buffer for overflow

```go
// Example: Disk-backed overflow buffer
type HybridBuffer struct {
    memory     *RingBuffer
    disk       *DiskBuffer
    threshold  float64 // e.g., 0.8 = use disk when 80% full
}
```

#### 2. Slow Serialization Performance
**Symptoms**: Low experience collection rate, game lag

**Solutions**:
- Enable memory pooling
- Parallelize serialization for multiple players
- Profile and optimize hot paths
- Consider reducing tensor precision (float32 → float16)

```bash
# Profile serialization performance
go test -bench=BenchmarkStateToTensor -cpuprofile=cpu.prof
go tool pprof cpu.prof
```

#### 3. Network Bottlenecks
**Symptoms**: Trainer lag, dropped experiences, timeouts

**Solutions**:
- Increase batch sizes to reduce message overhead
- Enable compression for network transfers
- Implement local caching for trainers
- Use multiple gRPC connections for parallelism

```yaml
# config/experience.yaml
streaming:
  batch_size: 256  # Increase from default 32
  compression: true
  max_message_size: 10485760  # 10MB
```

#### 4. Experience Loss/Corruption
**Symptoms**: Missing experiences, training instability

**Solutions**:
- Enable write-ahead logging
- Implement checksums for experience validation
- Add retry logic for failed writes
- Monitor experience sequence numbers

```go
// Example: Experience validation
func ValidateExperience(exp *Experience) error {
    if exp.State == nil || exp.NextState == nil {
        return errors.New("missing state data")
    }
    if exp.Action < 0 || exp.Action >= MaxActions {
        return errors.New("invalid action")
    }
    if math.IsNaN(float64(exp.Reward)) {
        return errors.New("NaN reward")
    }
    return nil
}
```

#### 5. Storage Write Failures
**Symptoms**: S3 errors, disk full, permission denied

**Solutions**:
- Implement exponential backoff retry
- Use multi-destination writes
- Monitor disk usage and alert early
- Rotate/archive old experiences

```go
// Example: Retry with backoff
func WriteWithRetry(writer Writer, data []byte) error {
    backoff := 100 * time.Millisecond
    for i := 0; i < 5; i++ {
        err := writer.Write(data)
        if err == nil {
            return nil
        }
        
        log.Warn().Err(err).Int("attempt", i+1).Msg("Write failed, retrying")
        time.Sleep(backoff)
        backoff *= 2
    }
    return fmt.Errorf("write failed after 5 attempts")
}
```

### Monitoring Checklist

1. **System Metrics**:
   - CPU usage per component
   - Memory usage and growth rate
   - Network bandwidth utilization
   - Disk I/O and space usage

2. **Application Metrics**:
   - Experiences collected per second
   - Serialization latency (p50, p95, p99)
   - Buffer fill rate and drops
   - gRPC stream health and latency

3. **Data Quality Metrics**:
   - Reward distribution statistics
   - Action distribution (ensure exploration)
   - Experience validation failures
   - Compression ratios achieved

### Debug Commands

```bash
# Check experience collection rate
curl -s localhost:9090/metrics | grep experiences_collected_total

# Monitor buffer usage
watch -n 1 'curl -s localhost:9090/metrics | grep experience_buffer'

# Test gRPC streaming
grpcurl -plaintext localhost:50051 generals.experience.v1.ExperienceService/StreamExperiences

# Analyze memory usage
go tool pprof http://localhost:6060/debug/pprof/heap

# Check for goroutine leaks
curl http://localhost:6060/debug/pprof/goroutine?debug=2
```

## Future Enhancements

1. **Prioritized Experience Replay**: Weight experiences by TD-error
2. **Distributed Experience Buffer**: Redis/Hazelcast for shared buffer
3. **Experience Augmentation**: Generate synthetic experiences
4. **Curriculum Learning**: Adjust game difficulty based on agent performance
5. **Multi-Agent Experience Sharing**: Share experiences between similar agents
6. **Experience Deduplication**: Prevent duplicate experiences in distributed settings
7. **Adaptive Compression**: Adjust compression based on network/CPU trade-offs