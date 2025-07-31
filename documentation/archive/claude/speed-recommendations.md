# Production-Scale RL Training: Implementation Plan

This document outlines specific optimizations and features to implement for high-performance, production-scale reinforcement learning training with the Generals game engine.

## 1. Batch Game Management System

### 1.1 Game Pool Manager
**File:** `internal/game/pool/manager.go`

```go
type GamePool struct {
    games      map[string]*GameInstance
    available  chan string  // Pool of available game IDs
    maxGames   int
    mu         sync.RWMutex
}

type BatchGameRequest struct {
    NumGames     int
    GameConfig   Config
    NumAgents    int
    GameIDs      []string // Pre-allocated game IDs
}
```

**Implementation:**
- Pre-allocate N game instances at startup
- Reuse game instances between episodes (reset instead of recreate)
- Batch API endpoint: `CreateBatchGames(BatchGameRequest) BatchGameResponse`
- Connection pooling for agent-to-game assignments

### 1.2 Vectorized Game State API
**File:** `proto/game/v1/batch.proto`

```protobuf
service BatchGameService {
  // Get states for multiple games in a single call
  rpc GetBatchStates(BatchStateRequest) returns (BatchStateResponse);
  
  // Submit actions for multiple games
  rpc SubmitBatchActions(BatchActionRequest) returns (BatchActionResponse);
  
  // Reset multiple games for new episodes
  rpc ResetBatchGames(BatchResetRequest) returns (BatchResetResponse);
}

message BatchStateRequest {
  repeated GameStateQuery queries = 1;
}

message GameStateQuery {
  string game_id = 1;
  int32 player_id = 2;
}
```

**Benefits:**
- Reduce gRPC overhead with batched calls
- Enable vectorized numpy operations on Python side
- Amortize serialization costs

## 2. Performance Optimizations

### 2.1 Memory Pool for Common Objects
**File:** `internal/game/pool/memory.go`

```go
var (
    tilePool = sync.Pool{
        New: func() interface{} {
            return &Tile{}
        },
    }
    actionPool = sync.Pool{
        New: func() interface{} {
            return &MoveAction{}
        },
    }
    boardPool = sync.Pool{
        New: func() interface{} {
            return &Board{
                Tiles: make([]Tile, 0, 400), // Pre-size for 20x20
            }
        },
    }
)
```

**Implementation:**
- Pool frequently allocated objects (Tiles, Actions, Boards)
- Pre-allocate slices with capacity hints
- Reuse coordinate structs

### 2.2 Lock-Free Data Structures
**File:** `internal/game/concurrent/lockfree.go`

```go
// Lock-free ring buffer for action queue
type ActionRingBuffer struct {
    buffer   [1024]Action
    head     atomic.Uint64
    tail     atomic.Uint64
}

// Wait-free statistics counter
type StatsCounter struct {
    gamesPlayed    atomic.Uint64
    turnsProcessed atomic.Uint64
    actionsPerSec  atomic.Uint64
}
```

**Benefits:**
- Eliminate lock contention in hot paths
- Enable true parallel game processing
- Reduce context switching overhead

### 2.3 SIMD Optimizations for Board Operations
**File:** `internal/game/simd/board_ops.go`

```go
// Use SIMD instructions for visibility calculations
func UpdateVisibilityMaskSIMD(board *Board, playerID int) {
    // Process 8 tiles at once using AVX2
    // Requires build tags and assembly
}

// Vectorized legal move calculation
func CalculateLegalMovesSIMD(board *Board, playerID int) []bool {
    // Process multiple tiles in parallel
}
```

### 2.4 Zero-Copy Serialization
**File:** `internal/game/serialize/zero_copy.go`

```go
// Custom binary format for ultra-fast serialization
type CompactGameState struct {
    // Fixed-size representation
    Turn      uint32
    Width     uint8
    Height    uint8
    TileData  [400]CompactTile // Max 20x20
}

type CompactTile struct {
    TypeAndOwner uint8  // 4 bits type, 4 bits owner
    ArmyCount    uint16
}

// Direct memory mapping for IPC
func (g *GameState) MarshalBinary() ([]byte, error) {
    // Return slice backed by mmap'd memory
}
```

## 3. Monitoring and Metrics System

### 3.1 Prometheus Metrics
**File:** `internal/metrics/game_metrics.go`

```go
var (
    GamesActive = promauto.NewGauge(prometheus.GaugeOpts{
        Name: "generals_games_active",
        Help: "Number of active games",
    })
    
    TurnsProcessed = promauto.NewCounter(prometheus.CounterOpts{
        Name: "generals_turns_processed_total",
        Help: "Total number of turns processed",
    })
    
    ActionProcessingTime = promauto.NewHistogram(prometheus.HistogramOpts{
        Name:    "generals_action_processing_seconds",
        Help:    "Time to process a single action",
        Buckets: prometheus.ExponentialBuckets(0.00001, 2, 15), // 10µs to 327ms
    })
    
    GameStateSize = promauto.NewHistogram(prometheus.HistogramOpts{
        Name:    "generals_game_state_bytes",
        Help:    "Size of serialized game state",
        Buckets: prometheus.ExponentialBuckets(100, 2, 10), // 100B to 51KB
    })
)
```

### 3.2 Distributed Tracing
**File:** `internal/tracing/game_tracing.go`

```go
func (e *Engine) RunWithTracing(ctx context.Context) error {
    ctx, span := otel.Tracer("generals").Start(ctx, "game.run")
    defer span.End()
    
    span.SetAttributes(
        attribute.Int("game.players", len(e.state.Players)),
        attribute.Int("game.turn", e.state.Turn),
    )
    
    // Trace each phase
    ctx, actionSpan := otel.Tracer("generals").Start(ctx, "game.process_actions")
    e.processActions(ctx)
    actionSpan.End()
    
    // Add trace context to gRPC metadata
}
```

### 3.3 Performance Dashboard
**File:** `monitoring/grafana/dashboards/rl_training.json`

Key metrics to track:
- Games per second
- Actions per second  
- Turn processing time (p50, p95, p99)
- Memory usage per game
- gRPC request latency
- Agent connection pool status
- GPU utilization (if using GPU inference)

## 4. Horizontal Scaling Architecture

### 4.1 Game Server Sharding
**File:** `internal/shard/manager.go`

```go
type ShardManager struct {
    shards    []*GameShard
    hash      ConsistentHash
    replicas  int
}

type GameShard struct {
    id        int
    games     map[string]*Game
    rpcServer *grpc.Server
    port      int
}

// Route games to shards based on game ID
func (sm *ShardManager) GetShard(gameID string) *GameShard {
    return sm.shards[sm.hash.Get(gameID)]
}
```

### 4.2 Redis-Based Game State Cache
**File:** `internal/cache/redis_cache.go`

```go
type GameStateCache struct {
    client *redis.Client
    ttl    time.Duration
}

func (c *GameStateCache) SetGameState(gameID string, state []byte) error {
    return c.client.Set(ctx, fmt.Sprintf("game:%s", gameID), 
                       state, c.ttl).Err()
}

// Use Redis Streams for action queue
func (c *GameStateCache) PublishAction(gameID string, action Action) error {
    return c.client.XAdd(ctx, &redis.XAddArgs{
        Stream: fmt.Sprintf("actions:%s", gameID),
        Values: map[string]interface{}{
            "player_id": action.PlayerID,
            "data":      action.Marshal(),
        },
    }).Err()
}
```

## 5. Training-Specific Optimizations

### 5.1 Experience Buffer Service
**File:** `internal/rl/experience_buffer.go`

```go
type ExperienceBuffer struct {
    capacity   int
    buffer     []Experience
    priorities []float32  // For prioritized replay
    mu         sync.RWMutex
}

type Experience struct {
    State      CompactGameState
    Action     Action
    Reward     float32
    NextState  CompactGameState
    Done       bool
    GameID     string
    PlayerID   int
}

// Efficient sampling for training
func (eb *ExperienceBuffer) SampleBatch(size int) []Experience {
    // Use weighted sampling based on priorities
}
```

### 5.2 Checkpoint System
**File:** `internal/rl/checkpoint.go`

```go
type GameCheckpoint struct {
    GameID    string
    State     *GameState
    History   []Action
    Timestamp time.Time
}

type CheckpointManager struct {
    storage   CheckpointStorage
    interval  int // Checkpoint every N turns
}

// Save games for later analysis or resume
func (cm *CheckpointManager) SaveCheckpoint(game *Game) error {
    if game.Turn % cm.interval == 0 {
        return cm.storage.Save(game.ToCheckpoint())
    }
    return nil
}
```

## 6. Development Workflow Improvements

### 6.1 Performance Benchmarking Suite
**File:** `benchmarks/rl_benchmarks_test.go`

```go
func BenchmarkSelfPlay(b *testing.B) {
    benchmarks := []struct {
        name      string
        numGames  int
        numAgents int
    }{
        {"1_game_2_agents", 1, 2},
        {"10_games_2_agents", 10, 2},
        {"100_games_4_agents", 100, 4},
        {"1000_games_2_agents", 1000, 2},
    }
    
    for _, bm := range benchmarks {
        b.Run(bm.name, func(b *testing.B) {
            // Measure games/sec, actions/sec, memory usage
        })
    }
}
```

### 6.2 Load Testing Framework
**File:** `testing/load/rl_load_test.go`

```go
type LoadTestConfig struct {
    NumAgents       int
    GamesPerAgent   int
    ActionsPerSec   int
    Duration        time.Duration
    ThinkTime       time.Duration
}

func RunLoadTest(config LoadTestConfig) LoadTestResults {
    // Simulate realistic RL training load
    // Measure latencies, throughput, errors
}
```

## Implementation Priority

1. **Phase 1 (Week 1-2)**: Batch Game Management
   - Implement GamePool and batch APIs
   - Add vectorized state retrieval
   - Basic benchmarking

2. **Phase 2 (Week 3-4)**: Core Performance
   - Memory pooling
   - Zero-copy serialization  
   - Profile and optimize hot paths

3. **Phase 3 (Week 5-6)**: Monitoring
   - Prometheus metrics
   - Grafana dashboards
   - Basic distributed tracing

4. **Phase 4 (Week 7-8)**: Horizontal Scaling
   - Sharding system
   - Redis integration
   - Load balancing

5. **Phase 5 (Week 9-10)**: RL-Specific Features
   - Experience buffer
   - Checkpoint system
   - Training metrics

## Expected Performance Gains

With these optimizations, expect:
- **10-100x** increase in games/second
- **Sub-millisecond** action processing latency
- **Linear scaling** up to 10,000 concurrent games
- **50-80%** reduction in memory usage per game
- **Minimal GC pressure** even under high load

## Testing the Optimizations

Run benchmarks before and after each optimization:

```bash
# Baseline benchmark
go test -bench=. -benchmem -benchtime=10s ./benchmarks

# Profile CPU usage
go test -bench=. -cpuprofile=cpu.prof ./benchmarks
go tool pprof cpu.prof

# Profile memory allocations  
go test -bench=. -memprofile=mem.prof ./benchmarks
go tool pprof -alloc_space mem.prof

# Continuous profiling in production
# Use tools like Pyroscope or Parca
```

## Conclusion

These optimizations will transform your game engine into a high-performance RL training platform capable of:
- Running millions of game steps per second
- Supporting thousands of concurrent agents
- Scaling horizontally across multiple machines
- Providing detailed performance insights
- Maintaining sub-millisecond latencies under load

Start with Phase 1 (Batch Game Management) as it provides the foundation for all other optimizations and will immediately improve training throughput.