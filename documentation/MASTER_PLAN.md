# MASTER PLAN - GeneralsReinforcementLearning

## Executive Summary

This document consolidates all implementation plans into a single roadmap. The project aims to create a Generals.io-style game optimized for reinforcement learning research, supporting thousands of concurrent games for agent training.

### Current State
- ✅ Core game engine fully functional
- ✅ Event-driven architecture implemented (Phase 1)
- ✅ State machine for game lifecycle (Phase 2)
- ✅ Basic gRPC server with multiplayer support
- ✅ Ebiten UI client for visualization
- 🚧 StreamGame gRPC method (exists but not implemented)
- ❌ Python agents cannot play full games yet
- ❌ No multi-game support
- ❌ Experience collection not connected to gRPC

### Primary Objectives
1. Enable RL agents to play and learn from the game
2. Support massive parallel game execution (1000+ concurrent games)
3. Create efficient experience collection pipeline
4. Build distributed training infrastructure

## Completed Work

### Phase 1: Event-Driven Architecture ✅ (Completed July 2025)
- Comprehensive event system with 15+ event types
- Synchronous event bus implementation
- Integration with game engine and action processor
- Structured logging subscriber with filtering

### Phase 2: State Machine ✅ (Completed July 2025)
- 9-phase state machine (Initializing → Lobby → Starting → Running → Paused/Ending → Ended)
- Pause/resume functionality with time tracking
- State transition validation and history
- Single-player support for RL training

### Additional Completed Features
- Fog of war system with incremental updates
- Map generation with cities and mountains
- Turn-based gameplay with timeouts
- Docker containerization (basic)
- Unit tests for core components

## Implementation Phases

### Phase 1: Core Infrastructure (1-2 weeks) 🎯 CURRENT FOCUS

**Goal**: Get Python agents playing complete games

- [ ] **Complete StreamGame gRPC Implementation**
  - [ ] Server-side streaming for real-time updates
  - [ ] Event types for state changes
  - [ ] Handle multiple concurrent streams
  - [ ] Integration tests

- [ ] **Fix Python Agent Infrastructure**
  - [ ] Refactor base_agent.py following recommendations
  - [ ] Implement proper game loop with streaming
  - [ ] Add connection retry logic
  - [ ] Create working random_agent.py

- [ ] **Enable Full Game Playing**
  - [ ] Agents can join and start games
  - [ ] Receive turn notifications via stream
  - [ ] Submit valid moves
  - [ ] Handle game end conditions

- [ ] **Basic Monitoring**
  - [ ] Structured logging for debugging
  - [ ] Simple metrics (games/second, moves/turn)
  - [ ] Health check endpoints

### Phase 2: Multi-Game Support (1-2 weeks)

**Goal**: Run many games in parallel on single machine

- [ ] **Game Instance Manager**
  - [ ] Manage multiple concurrent game instances
  - [ ] Resource pooling and limits
  - [ ] Game lifecycle management
  - [ ] Performance optimization

- [ ] **Basic Matchmaking**
  - [ ] Queue management for waiting agents
  - [ ] Simple pairing logic (2-player games)
  - [ ] Auto-start when players ready
  - [ ] Handle disconnections

- [ ] **Parallel Execution**
  - [ ] Run 10+ games concurrently
  - [ ] Independent game loops
  - [ ] Efficient resource usage
  - [ ] No interference between games

- [ ] **Performance Benchmarking**
  - [ ] Measure games per second
  - [ ] CPU/memory profiling
  - [ ] Identify bottlenecks
  - [ ] Optimization targets

### Phase 3: Experience Collection (1-2 weeks)

**Goal**: Capture game data for RL training

- [ ] **Complete Experience Collector**
  - [ ] Fix existing implementation issues
  - [ ] State to tensor conversion
  - [ ] Action recording with context
  - [ ] Reward calculation

- [ ] **gRPC Integration**
  - [ ] StreamExperiences endpoint
  - [ ] Batch experience transmission
  - [ ] Filtering by game/player
  - [ ] Backpressure handling

- [ ] **Replay Buffer**
  - [ ] Circular buffer implementation
  - [ ] Prioritized sampling support
  - [ ] Memory-efficient storage
  - [ ] Configurable size limits

- [ ] **Storage Backend**
  - [ ] Local file storage option
  - [ ] S3 integration (optional)
  - [ ] Compression support
  - [ ] Replay functionality

### Phase 4: Basic RL Training (2-3 weeks)

**Goal**: Train an agent that improves beyond random play

- [ ] **Environment Wrapper**
  - [ ] OpenAI Gym/Gymnasium compatible
  - [ ] Observation space design
  - [ ] Action space mapping
  - [ ] Reset and step functions

- [ ] **Simple RL Agent**
  - [ ] Choose algorithm (DQN/PPO/etc)
  - [ ] Basic neural network
  - [ ] Training loop
  - [ ] Checkpointing

- [ ] **Training Pipeline**
  - [ ] Connect to experience stream
  - [ ] Batch processing
  - [ ] Model updates
  - [ ] Validation games

- [ ] **Metrics & Monitoring**
  - [ ] Training progress tracking
  - [ ] Win rate over time
  - [ ] TensorBoard integration
  - [ ] Performance metrics

### Phase 5: Self-Play & Evaluation (2-3 weeks)

**Goal**: Enable agents to train against themselves and track progress

- [ ] **Model Management**
  - [ ] Save/load model checkpoints
  - [ ] Version tracking
  - [ ] Model serving endpoint
  - [ ] Hot-swapping support

- [ ] **Self-Play Infrastructure**
  - [ ] Agent vs historical versions
  - [ ] Symmetric game setup
  - [ ] Experience from both perspectives
  - [ ] Automatic matchmaking

- [ ] **Evaluation System**
  - [ ] ELO rating implementation
  - [ ] Win/loss tracking
  - [ ] Performance statistics
  - [ ] Progress visualization

- [ ] **Tournament Support**
  - [ ] Round-robin tournaments
  - [ ] Bracket generation
  - [ ] Result tracking
  - [ ] Leaderboards

### Phase 6: Distributed Training (3-4 weeks)

**Goal**: Scale to thousands of concurrent games across multiple machines

- [ ] **Multi-Node Architecture**
  - [ ] Game server clustering
  - [ ] Service discovery
  - [ ] Load balancing
  - [ ] Fault tolerance

- [ ] **Distributed Games**
  - [ ] Remote game creation
  - [ ] Cross-node matchmaking
  - [ ] Network optimization
  - [ ] Latency handling

- [ ] **Centralized Experience**
  - [ ] Experience aggregation service
  - [ ] Distributed replay buffer
  - [ ] Efficient data transfer
  - [ ] Deduplication

- [ ] **Scalable Training**
  - [ ] Multiple learner nodes
  - [ ] Distributed model updates
  - [ ] Gradient aggregation
  - [ ] Synchronization

### Phase 7: Production Features (Ongoing)

**Goal**: Optimize and enhance the system for research use

- [ ] **Advanced Features**
  - [ ] Curriculum learning
  - [ ] Population-based training
  - [ ] League play
  - [ ] Meta-learning

- [ ] **Optimizations**
  - [ ] GPU acceleration
  - [ ] Vectorized environments
  - [ ] JIT compilation
  - [ ] Memory pooling

- [ ] **Research Tools**
  - [ ] Experiment tracking
  - [ ] Hyperparameter search
  - [ ] A/B testing framework
  - [ ] Analysis tools

- [ ] **Documentation**
  - [ ] API documentation
  - [ ] Training guides
  - [ ] Architecture docs
  - [ ] Research papers

## Success Metrics

### Performance Targets
- **Local**: 100+ concurrent games on single machine
- **Distributed**: 1000+ concurrent games across cluster
- **Latency**: <100ms action processing
- **Throughput**: 50k+ experiences/second

### Training Targets
- Agent surpasses random play within 1 hour
- Reaches competent human-level play within 24 hours
- Stable training without divergence
- Consistent improvement over generations

### System Reliability
- 99.9% uptime for training runs
- Automatic recovery from failures
- No data loss during crashes
- Deterministic replay capability

## Next Immediate Actions

1. **Complete StreamGame Implementation** (Highest Priority)
   - This is blocking all Python agent development
   - Without it, agents can't play effectively

2. **Fix Python Random Agent**
   - Refactor base_agent.py
   - Create functional random agent
   - Test full game completion

3. **Run First Multi-Agent Games**
   - Two random agents playing
   - Verify complete game flow
   - Collect basic metrics

4. **Begin Multi-Game Support**
   - Start with 2-3 concurrent games
   - Gradually scale up
   - Profile performance

## Related Documentation

- **RL Details**: See `documentation/rl-training.md` for reward functions, neural network architectures
- **Deployment**: See `documentation/deployment-architecture.md` for AWS/Terraform setup
- **Architecture**: See `documentation/architecture/` for detailed system design
- **Python Agents**: See `documentation/python-agent-architecture.md` for agent implementation details

## Notes

- Each phase builds on the previous one
- Phases can overlap slightly but dependencies must be respected
- Focus on getting basics working before adding complexity
- Regular testing and validation at each phase
- Document issues and solutions as they arise

---

*Last Updated: 2025-07-31*
*This is the authoritative planning document. All other plans are superseded by this one.*