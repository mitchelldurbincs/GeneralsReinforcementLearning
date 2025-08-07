# End-to-End Streaming and Multi-Game Support Implementation Plan

## Executive Summary

This document outlines the implementation plan for optimizing the GeneralsReinforcementLearning system to support efficient end-to-end experience streaming and scale to thousands of concurrent games for reinforcement learning training.

**Current State**: The system already supports multiple concurrent games with streaming, but resource usage patterns limit practical scaling to ~500-1000 games per server.

**Target State**: Support 2000-5000 concurrent games per server with efficient experience streaming to Python RL agents.

## Current Architecture Analysis

### Strengths ✅
1. **StreamGame**: Production-ready implementation with delta compression and fog-of-war filtering
2. **Experience Collection**: Well-designed with per-game buffers and proper interfaces
3. **Game Isolation**: Proper separation between game instances with no shared state
4. **Python Integration**: Async streaming with tensor conversion already implemented

### Bottlenecks 🚧
1. **Goroutine Proliferation**: Each game spawns 2-3 goroutines (experience polling, turn timers)
2. **Memory Overhead**: Per-game stream managers with individual state tracking
3. **Polling Inefficiency**: 100ms polling intervals for experience collection
4. **Cleanup Overhead**: Global 5-minute scans with potential lock contention

## Implementation Phases

### Phase 1: Optimize Resource Usage (Priority: HIGH)

#### 1.1 Centralized Experience Collection Service ✅ COMPLETED

**Problem**: Each game creates a goroutine polling experiences every 100ms

**Solution**: Single service managing all game experience collection

**TODOs:**
- [x] Create `internal/grpc/gameserver/experience_aggregator.go`
  ```go
  type ExperienceAggregator struct {
      games    map[string]*ExperienceCollector
      mu       sync.RWMutex
      updateCh chan string // game IDs needing collection
  }
  ```
- [x] Implement single goroutine processing experience updates
- [x] Replace per-game polling with event-driven collection
- [x] Add batching for efficient processing (collect from multiple games per cycle)
- [x] Update `game_manager.go` to register games with aggregator

**Files Modified:**
- ✅ `internal/grpc/gameserver/game_manager.go` - Updated to use aggregator
- ✅ `internal/grpc/gameserver/experience_aggregator.go` - New centralized service
- ✅ `internal/experience/collector.go` - Added GetCount() method

**Implementation Notes:**
- Reduced goroutines from O(games) to 1 for experience collection
- Event-driven collection with notification system
- Batch processing with configurable intervals (50ms default)
- Automatic cleanup when games are unregistered
- Graceful shutdown with final flush of all experiences

#### 1.2 Timer Wheel for Turn Management

**Problem**: Each game creates individual time.Timer goroutines

**Solution**: Centralized timer wheel managing all game turn timeouts

**TODOs:**
- [ ] Create `internal/grpc/gameserver/timer_wheel.go`
  ```go
  type TimerWheel struct {
      slots     []*list.List
      ticker    *time.Ticker
      precision time.Duration
  }
  ```
- [ ] Implement hierarchical timing wheel (100ms precision)
- [ ] Replace individual turn timers with wheel registration
- [ ] Handle timer cancellation and rescheduling
- [ ] Add metrics for timer efficiency

**Files to Modify:**
- `internal/game/engine.go` (turn timeout handling)
- `internal/grpc/gameserver/server.go` (lines 450-500)

#### 1.3 Memory Pool for Stream Managers

**Problem**: Creating new stream managers for each game

**Solution**: Pre-allocated pool of reusable stream managers

**TODOs:**
- [ ] Create `internal/grpc/gameserver/stream_pool.go`
- [ ] Implement sync.Pool for StreamManager instances
- [ ] Add reset method to StreamManager for reuse
- [ ] Update game creation to use pooled managers
- [ ] Add metrics for pool efficiency

### Phase 2: Scale Multi-Game Support (Priority: HIGH)

#### 2.1 Game Sharding System

**Problem**: Single map holding all games with global lock

**Solution**: Sharded game storage with per-shard locks

**TODOs:**
- [ ] Implement sharded game storage in `game_manager.go`
  ```go
  type GameShard struct {
      games map[string]*GameInstance
      mu    sync.RWMutex
  }
  
  type ShardedGameManager struct {
      shards []*GameShard
      // Use consistent hashing for shard selection
  }
  ```
- [ ] Add consistent hashing for game ID → shard mapping
- [ ] Implement per-shard cleanup routines
- [ ] Add shard rebalancing for even distribution
- [ ] Update all game operations to use sharding

#### 2.2 Batch Experience Streaming

**Problem**: Individual experience streams per client

**Solution**: Multiplexed streaming with client routing

**TODOs:**
- [ ] Create experience routing service
- [ ] Implement client subscription management
- [ ] Add experience batching per stream
- [ ] Optimize protobuf message sizes
- [ ] Add compression for large batches

### Phase 3: Production Optimization (Priority: MEDIUM)

#### 3.1 Incremental Cleanup System

**Problem**: Full scan of all games every 5 minutes

**Solution**: Priority queue-based incremental cleanup

**TODOs:**
- [ ] Implement priority queue for game cleanup times
- [ ] Process only expired games
- [ ] Add graceful cleanup with player notification
- [ ] Implement cleanup backpressure handling

#### 3.2 Resource Monitoring

**TODOs:**
- [ ] Add Prometheus metrics for:
  - Games per second created/destroyed
  - Experience throughput (experiences/second)
  - Goroutine count per game
  - Memory usage per game
  - Stream bandwidth usage
- [ ] Create Grafana dashboard for monitoring
- [ ] Add alerts for resource thresholds

### Phase 4: Testing and Validation (Priority: HIGH)

#### 4.1 Load Testing Suite

**TODOs:**
- [ ] Create `tests/load/multi_game_test.go`
- [ ] Implement test scenarios:
  - Ramp up to 1000 games
  - Sustained 2000 games for 1 hour
  - Burst creation/destruction patterns
  - Experience streaming under load
- [ ] Add performance regression tests
- [ ] Create benchmarking suite

#### 4.2 Integration Testing

**TODOs:**
- [ ] Create end-to-end test with Python agents
- [ ] Test experience collection accuracy
- [ ] Validate stream reliability under load
- [ ] Test graceful degradation scenarios

## Configuration Changes

### Server Configuration (`config/server.yaml`)
```yaml
game:
  max_concurrent_games: 5000
  game_shards: 16
  experience_batch_size: 100
  experience_flush_interval: 50ms
  
resources:
  timer_wheel_precision: 100ms
  stream_pool_size: 1000
  max_goroutines_per_game: 2
  
monitoring:
  metrics_enabled: true
  metrics_port: 9090
```

### Python Client Configuration
```python
# generals_agent/config.py
STREAMING_CONFIG = {
    'batch_timeout': 0.1,  # seconds
    'max_batch_size': 100,
    'experience_buffer_size': 10000,
    'stream_reconnect_delay': 1.0
}
```

## Performance Targets

### Current Performance
- **Concurrent Games**: 500-1000
- **Goroutines per Game**: 3-4
- **Memory per Game**: ~5MB
- **Experience Latency**: 100-200ms

### Target Performance
- **Concurrent Games**: 2000-5000
- **Goroutines per Game**: <0.5 (shared services)
- **Memory per Game**: ~2MB
- **Experience Latency**: <50ms

## Migration Strategy

1. **Week 1**: Implement Phase 1 optimizations
   - No breaking changes
   - Gradual rollout with feature flags
   - Monitor performance improvements

2. **Week 2**: Deploy Phase 2 scaling improvements
   - Test with increasing game loads
   - Validate Python client compatibility

3. **Week 3**: Production optimization and monitoring
   - Full monitoring deployment
   - Performance validation

## Risk Mitigation

1. **Backwards Compatibility**: All changes maintain API compatibility
2. **Gradual Rollout**: Feature flags for each optimization
3. **Rollback Plan**: Version tagging for quick reversion
4. **Testing**: Comprehensive load testing before production

## Success Metrics

- [ ] Support 2000+ concurrent games on 4-core server
- [ ] Maintain <50ms experience collection latency
- [ ] Reduce goroutine count by 90%
- [ ] Achieve 200k+ experiences/second throughput
- [ ] Zero experience loss under load

## Appendix: Code Examples

### Example: Centralized Experience Aggregator
```go
func (ea *ExperienceAggregator) Start() {
    go func() {
        ticker := time.NewTicker(10 * time.Millisecond)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                ea.collectBatch()
            case gameID := <-ea.updateCh:
                ea.markForCollection(gameID)
            }
        }
    }()
}
```

### Example: Python Batch Consumer
```python
async def consume_experience_batches(self):
    async for batch in self.experience_stream:
        experiences = [
            self._parse_experience(exp) 
            for exp in batch.experiences
        ]
        await self.experience_buffer.extend(experiences)
```

## Next Steps

1. Review and approve implementation plan
2. Create feature branches for each phase
3. Begin Phase 1 implementation
4. Set up performance monitoring baseline
5. Schedule weekly progress reviews

---

**Document Version**: 1.1  
**Last Updated**: 2025-08-03  
**Author**: Claude  
**Status**: Phase 1.1 Complete - Experience Aggregator Implemented

## Implementation Progress

### Completed ✅
- **Phase 1.1**: Centralized Experience Collection Service
  - Reduced per-game goroutines from 100ms polling to single aggregator
  - Event-driven collection with batching
  - Estimated goroutine reduction: 90%+ for experience collection

### Next Steps
- **Phase 1.2**: Timer Wheel for Turn Management (HIGH priority)
- **Phase 1.3**: Memory Pool for Stream Managers (MEDIUM priority)