# Architecture Recommendations for GeneralsReinforcementLearning

This document outlines architectural improvements to transform the current game engine into a more extensible, maintainable, and RL-training-optimized platform. The recommendations focus on two primary patterns: Event-Driven Architecture and State Management, along with supporting infrastructure for reinforcement learning.

## Overview

The current codebase is well-structured but tightly coupled in some areas. These recommendations will:
- Decouple components through event-driven patterns
- Formalize game state management with a state machine
- Prepare infrastructure for scalable RL training
- Improve debugging and monitoring capabilities

## Phase 1: Event-Driven Architecture Foundation (2-3 weeks)

### Objective
Introduce an event bus system to decouple game logic from auxiliary concerns like metrics, logging, and RL data collection.

### Implementation Tasks

- [ ] Create event system foundation
  - [ ] Define base event interfaces in `internal/game/events/types.go`
  - [ ] Implement core event types (GameStarted, TurnStarted, ActionProcessed, etc.)
  - [ ] Create EventBus with synchronous dispatch in `internal/game/events/bus.go`
  - [ ] Add event timestamp and metadata support

- [ ] Define game events
  - [ ] Movement events (MoveSubmitted, MoveValidated, MoveExecuted)
  - [ ] Combat events (CombatStarted, CombatResolved, TilesCaptured)
  - [ ] Player events (PlayerJoined, PlayerEliminated, PlayerWon)
  - [ ] Production events (ProductionApplied, CityProduced, GeneralProduced)

- [ ] Integrate with existing engine
  - [ ] Add EventBus to Engine struct
  - [ ] Publish events from ProcessTurn without changing logic
  - [ ] Publish events from action processor
  - [ ] Ensure backward compatibility

- [ ] Create event logger subscriber
  - [ ] Implement structured event logging
  - [ ] Add event filtering by type
  - [ ] Create development-mode event dumper

### Code Structure
```
internal/game/events/
├── types.go          # Event interfaces and base types
├── bus.go            # EventBus implementation
├── game_events.go    # Specific game event definitions
└── subscribers/
    └── logger.go     # Logging subscriber
```

### Success Criteria
- [ ] Can subscribe to and receive game events
- [ ] No change to existing game behavior
- [ ] All major game actions publish events
- [ ] Event logging improves debugging

## Phase 2: State Machine Implementation (2-3 weeks)

### Objective
Formalize game flow with a proper state machine to prevent invalid operations and support clean episode management for RL.

### Implementation Tasks

- [ ] Create state machine framework
  - [ ] Define GamePhase enum in `internal/game/states/phases.go`
  - [ ] Implement transition rules and validation
  - [ ] Create state machine with enter/exit callbacks
  - [ ] Add state history tracking for debugging

- [ ] Define game phases
  - [ ] PhaseInitializing - Game object creation
  - [ ] PhaseLobby - Players joining, configuration
  - [ ] PhaseStarting - Map generation, player placement
  - [ ] PhaseRunning - Active gameplay
  - [ ] PhasePaused - Temporary suspension
  - [ ] PhaseEnding - Winner determination, cleanup
  - [ ] PhaseEnded - Final state
  - [ ] PhaseError - Error recovery state
  - [ ] PhaseReset - Reset the current game without full teardown

- [ ] Implement transition logic
  - [ ] Lobby → Starting (all players ready)
  - [ ] Starting → Running (setup complete)
  - [ ] Running → Paused (external trigger)
  - [ ] Paused → Running (resume)
  - [ ] Running → Ending (win condition)
  - [ ] Any → Error (unrecoverable error)

- [ ] Integrate with game engine
  - [ ] Add state machine to Engine
  - [ ] Guard action processing by state
  - [ ] Publish state transition events
  - [ ] Update gRPC endpoints to check state

- [ ] Add pause/resume support
  - [ ] Implement pause command
  - [ ] Save pause timestamp
  - [ ] Handle timeout during pause
  - [ ] Resume with state validation

### Code Structure
```
internal/game/states/
├── phases.go         # Phase definitions
├── machine.go        # State machine implementation
├── transitions.go    # Transition rules
└── context.go        # Game context for state decisions
```

### Success Criteria
- [ ] Clear game phase progression
- [ ] Invalid operations prevented by state
- [ ] Can pause/resume games
- [ ] State transitions are logged

## Phase 3: RL Infrastructure - Serialization (1-2 weeks)

### Objective
Implement efficient game state serialization for checkpointing and neural network input.

### Implementation Tasks

- [ ] Create serialization framework
  - [ ] Define StateSnapshot struct
  - [ ] Implement binary serialization with gob
  - [ ] Add compression support (gzip/zstd)
  - [ ] Create version handling for compatibility

- [ ] Implement tensor serialization
  - [ ] Convert board to multi-channel tensor
  - [ ] Normalize values for neural networks
  - [ ] Add player-relative views
  - [ ] Implement action mask generation

- [ ] Add checkpointing support
  - [ ] Save game state to disk
  - [ ] Load game state from checkpoint
  - [ ] Validate checkpoint integrity
  - [ ] Add checkpoint metadata

- [ ] Create state differ
  - [ ] Implement state comparison
  - [ ] Generate state deltas
  - [ ] Compress repeated states
  - [ ] Support incremental updates

### Code Structure
```
internal/game/serialization/
├── snapshot.go       # State snapshot types
├── serializer.go     # Serialization logic
├── tensor.go         # RL tensor conversion
└── checkpoint.go     # Checkpoint management
```

### Success Criteria
- [ ] Can serialize/deserialize game state
- [ ] Tensor format suitable for PyTorch/TensorFlow
- [ ] Checkpoint save/load works correctly
- [ ] Performance meets RL training needs

## Phase 4: Async Event Processing (1-2 weeks)

### Objective
Upgrade event bus to support asynchronous processing for better performance and scalability.

### Implementation Tasks

- [ ] Implement async event bus
  - [ ] Add buffered channel for events
  - [ ] Create worker pool for processing
  - [ ] Implement back-pressure handling
  - [ ] Add metrics for queue depth

- [ ] Create replay buffer collector
  - [ ] Subscribe to relevant events
  - [ ] Convert events to experiences
  - [ ] Implement circular buffer storage
  - [ ] Add sampling methods

- [ ] Add priority event handling
  - [ ] Define event priorities
  - [ ] Implement priority queue dispatch
  - [ ] Ensure critical events process first
  - [ ] Add timeout handling

- [ ] Implement event persistence
  - [ ] Save events to disk for replay
  - [ ] Create event replay system
  - [ ] Add event filtering/querying
  - [ ] Support event stream export

### Success Criteria
- [ ] Async processing doesn't block game
- [ ] Can handle high event throughput
- [ ] Replay buffer fills automatically
- [ ] Event persistence enables debugging

## Phase 5: Metrics and Monitoring (1-2 weeks)

### Objective
Build comprehensive metrics collection using the event system for training monitoring and debugging.

### Implementation Tasks

- [ ] Create metrics collector
  - [ ] Subscribe to all game events
  - [ ] Aggregate per-game metrics
  - [ ] Track per-player statistics
  - [ ] Calculate derived metrics

- [ ] Implement Prometheus integration
  - [ ] Export game counters
  - [ ] Add performance histograms
  - [ ] Track resource usage
  - [ ] Create custom RL metrics

- [ ] Add TensorBoard support
  - [ ] Export training metrics
  - [ ] Log game visualizations
  - [ ] Track win rates over time
  - [ ] Support custom scalars

- [ ] Create metrics dashboard
  - [ ] Design Grafana dashboards
  - [ ] Add alerting rules
  - [ ] Monitor system health
  - [ ] Track training progress

### Code Structure
```
internal/rl/metrics/
├── collector.go      # Metrics collection
├── prometheus.go     # Prometheus export
├── tensorboard.go    # TensorBoard export
└── aggregator.go     # Metric aggregation
```

### Success Criteria
- [ ] Comprehensive game metrics available
- [ ] Can monitor training in real-time
- [ ] Performance overhead < 5%
- [ ] Metrics help identify issues

## Phase 6: Training Pipeline Integration (2-3 weeks)

### Objective
Build complete training pipeline infrastructure connecting the Go engine to Python RL frameworks.

### Implementation Tasks

- [ ] Create experience generator
  - [ ] Parallel game simulation
  - [ ] Experience batching
  - [ ] Async experience collection
  - [ ] Memory-mapped storage

- [ ] Implement gRPC training service
  - [ ] Define training protobuf API
  - [ ] Create experience streaming
  - [ ] Add model checkpoint handling
  - [ ] Support distributed training

- [ ] Build Python client library
  - [ ] gRPC client wrapper
  - [ ] PyTorch data loader
  - [ ] TensorFlow dataset API
  - [ ] Utility functions

- [ ] Add training orchestration
  - [ ] Manage multiple game instances
  - [ ] Load balancing for games
  - [ ] Auto-scaling support
  - [ ] Fault tolerance

### Success Criteria
- [ ] Can train models using Go engine
- [ ] Supports distributed training
- [ ] Efficient data transfer
- [ ] Easy Python integration

## Implementation Guidelines

### Principles
1. **Incremental Adoption**: Each phase should work independently
2. **Backward Compatibility**: Don't break existing functionality
3. **Performance First**: Optimize for RL training throughput
4. **Testability**: Every component should be easily testable

### Testing Strategy
- Unit tests for each new component
- Integration tests for phase completion
- Performance benchmarks for critical paths
- End-to-end tests for full pipeline

### Migration Path
1. Start with read-only event subscribers
2. Gradually move logic to event-driven patterns
3. Introduce state machine guards incrementally
4. Migrate to async processing once stable

### Performance Targets
- Event processing: < 1μs overhead per event
- State serialization: < 1ms for typical game
- Tensor conversion: < 100μs per state
- Overall overhead: < 10% impact on game throughput

## Long-term Vision

This architecture enables:
- **Distributed Training**: Run thousands of games in parallel
- **A/B Testing**: Compare different game rules/parameters
- **Live Monitoring**: Watch training progress in real-time
- **Replay Analysis**: Deep dive into specific games
- **Model Versioning**: Track which rules/code produced which model
- **Multi-Agent Training**: Support for self-play and league play

## Next Steps

1. Review and approve architecture plan
2. Create detailed tickets for Phase 1
3. Set up development branch
4. Begin implementation of event system
5. Schedule regular architecture review meetings

## Questions to Resolve

- [ ] Should events be versioned for compatibility?
- [ ] What's the target game throughput for RL training?
- [ ] Should we support multiple event bus implementations?
- [ ] How should we handle event schema evolution?
- [ ] What metrics are most important for RL training?

---

*Document created by Claude on 2025-07-26*
*Based on analysis of GeneralsReinforcementLearning codebase*