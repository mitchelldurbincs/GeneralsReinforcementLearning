# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Important Note 1:** 
- If you are working from an implementation markdown file, using - [ ] to signify new tasks and mark your tasks completed with - [x] once you are done please. 

## Project Overview

This is a Go-based implementation of a Generals.io-style strategic territory control game designed specifically for reinforcement learning research and training. The game features army movement, territory capture, and fog of war mechanics. 

**Important Note**: This is NOT a multiplayer game server. While the codebase includes networking infrastructure (gRPC), it's intended for:
- Training reinforcement learning agents through self-play
- Running distributed RL training across multiple machines
- Occasional human play for testing agent performance
- Bot vs bot competitions in controlled environments

The project prioritizes RL training efficiency over production multiplayer features like security, authentication, or public-facing APIs.

### Game Mechanics
- **Turn-based strategy**: Players take turns moving armies
- **Territory control**: Capture tiles by moving armies onto them
- **Army production**: Generals and cities produce 1 army per turn
- **Combat**: Larger army wins when armies meet, difference remains
- **Victory conditions**: Capture all enemy generals or control entire map
- **Map features**: Mountains (impassable), cities (capturable for production)

## Development Commands

### Building
```bash
# Build the game server (headless)
go build -o game_server ./cmd/game_server

# Build the UI client (with graphics)
go build -o ui_client ./cmd/ui_client

# Build for Docker deployment (Note: cmd/game doesn't exist, use game_server)
CGO_ENABLED=0 go build -o game_server ./cmd/game_server
```

### Testing
```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run tests for a specific package
go test ./internal/game/core
go test ./internal/game/mapgen
go test ./internal/ui/renderer
go test ./internal/common
```

### Running
```bash
# Start the game server
go run cmd/game_server/main.go

# Start the UI client
go run cmd/ui_client/main.go

# Run Python agents (MUST activate virtual environment first)
source generalsrl/bin/activate
python python/test_grpc_client.py
python python/scripts/run_random_match.py

# Manage Go dependencies
go mod tidy
go mod download
```

## Architecture Overview

### Core Game Logic (`internal/game/`)
- **engine.go**: Main game loop, turn processing, action execution, win condition checking
  - Integrates with event system and state machine
  - Includes pause/resume functionality
  - Guards action processing based on game state
- **state.go**: Game state management, player and board state tracking
- **constants.go**: Game balance parameters (production rates, growth intervals)
- **visibility.go**: Fog of war implementation with incremental updates (3x3 visibility around owned tiles)
- **stats.go**: Player statistics tracking with optimized incremental updates
- **rendering.go**: Terminal-based board visualization with ANSI colors
- **demo_helpers.go**: Random action generation for testing and demonstrations
- **experience_collector.go**: Interface for collecting RL training experiences
- **core/**: Low-level game mechanics
  - `board.go`: 2D grid-based game board implementation (includes tile types: empty, mountain, city, general)
  - `movement.go`: Army movement and combat resolution
  - `coordinate.go`: Coordinate type with math operations and serialization
  - `action.go`: Action types and validation
  - `errors.go`: Domain-specific error types
  - `utils.go`: Core utility functions
- **mapgen/**: Procedural map generation
  - `generator.go`: Map generation with city and mountain placement
- **processor/**: Action processing
  - `action_processor.go`: Centralized action processing with event publishing
- **rules/**: Game rules and win conditions
  - `win_conditions.go`: Win condition checking
  - `legal_moves.go`: Legal move calculation for action masks
- **events/**: Event-driven architecture (Phase 1 completed)
  - `types.go`: Base event interfaces and types
  - `bus.go`: Synchronous event bus implementation
  - `game_events.go`: All game event definitions
  - `adapter.go`: Event publisher adapter for action processor
  - `subscribers/logger.go`: Structured logging subscriber
- **states/**: State machine implementation (Phase 2 partially completed)
  - `phases.go`: Game phase definitions (9 phases including Paused)
  - `machine.go`: State machine with transition rules and history
  - `states.go`: Individual state implementations
  - `context.go`: Game context for state decisions
  - Note: Single-player games are supported for RL training

### UI System (`internal/ui/`)
- **game.go**: Base game interface definition
- **human_game.go**: Human player implementation
- **renderer/**: Ebiten-based rendering system
  - `board.go`: Basic board rendering
  - `enhanced_board.go`: Enhanced visual features
  - `fog.go`: Fog of war rendering
  - `units.go`: Unit/army visualization
- **input/**: Input handling system
  - `handler.go`: Main input coordination
  - `mouse.go`: Mouse interactions
  - `keyboard.go`: Keyboard shortcuts
  - `actions.go`: Input action mapping

### Common Utilities (`internal/common/`)
- **colors.go**: Color definitions and utilities
- **math.go**: Mathematical helper functions
- **validation.go**: Coordinate validation and distance calculations

### Entry Points (`cmd/`)
- **game_server/**: gRPC server for hosting games and managing multiplayer sessions
- **ui_client/**: Graphical client for human players using Ebiten

### Protocol Buffers (`proto/`)
- **game/v1/**: Game service protocol definitions
- **experience/v1/**: Experience collection protocol definitions
- **common/v1/**: Shared message types

### Networking (`internal/grpc/`)
- **gameserver/**: gRPC server implementation
  - `server.go`: Main server logic, game management, player sessions
  - `stream_manager.go`: Real-time game state streaming
  - `game_manager.go`: Game lifecycle management
  - `experience_service.go`: Experience collection service
  - `experience_aggregator.go`: Aggregates experiences from multiple games
  - `converters.go`: Proto/internal type conversions
  - `action_validator.go`: Action validation logic

### Experience Collection (`internal/experience/`)
- **collector.go**: Base experience collector implementation
- **collector_optimized.go**: Performance-optimized collector
- **enhanced_collector.go**: Advanced collector with additional features
- **buffer.go**: Experience buffer for batching
- **lockfree_buffer.go**: Lock-free buffer for concurrent access
- **player_buffer_manager.go**: Per-player buffer management
- **rewards.go**: Reward calculation utilities
- **serializer.go**: Experience serialization
- **persistence.go**: Experience persistence to disk

### Python Integration (`python/`)
**Important**: Python code requires activating the virtual environment first:
```bash
source generalsrl/bin/activate
```
- **generals_agent/**: Core agent module
  - `base_agent.py`: Abstract base class for all agents
  - `random_agent.py`: Random action agent implementation
  - `game_client.py`: gRPC client wrapper
  - `game_session.py`: Game session management
  - `connection.py`: Connection handling
  - `experience_consumer.py`: Experience streaming consumer
  - `agent_runner.py`: Agent execution orchestration
  - `types.py`: Type definitions
  - `events.py`: Event handling
- **generals_pb/**: Generated protobuf files
- **examples/**: Example scripts and usage demonstrations
- **scripts/**: Utility scripts (e.g., `run_random_match.py`)

### Key Game Parameters
- City spawn ratio: 1 per 20 tiles (~5% of map)
- City starting army: 40 units
- Production: 1 army/turn for generals and cities
- Normal tile growth: Every 25 turns
- Fog of war: Fully implemented with toggle via config (`fog_of_war.enabled`)
- Minimum general spacing: 5 (Manhattan distance)
- Turn time limit: 0 (disabled) by default, configurable via `turn_timeout`
- Map sizes: Small (10x10), Medium (15x15), Large (20x20)
- Maximum concurrent games: 100 (configurable via `max_games`)

### Deployment
- Docker multi-stage builds for containerization (golang:1.24-alpine → alpine:3.21)
  - `deploy/Dockerfile`: Production deployment build
  - `docker/game-server/Dockerfile`: Alternative game server build
- Terraform configuration for AWS deployment (`deploy/terraform/`):
  - `compute_game_server.tf`, `compute_rl_trainer.tf`: EC2 instances
  - `ecr.tf`: Container registry
  - `s3.tf`: Storage
  - `networking.tf`, `security_groups.tf`: VPC and security
  - `iam.tf`: IAM roles and policies
  - `monitoring.tf`: CloudWatch configuration

## Event System Architecture

The game uses an event-driven architecture to decouple core game logic from auxiliary systems:

### Event Types
- **Game Lifecycle**: GameStarted, GameEnded, StateTransition
- **Turn Management**: TurnStarted, TurnEnded
- **Actions**: ActionSubmitted, ActionValidated, ActionExecuted, ActionRejected
- **Combat**: CombatStarted, CombatResolved, TilesCaptured, TilesLost
- **Player**: PlayerJoined, PlayerLeft, PlayerEliminated, PlayerWon
- **Production**: ProductionApplied, CityProduced, GeneralProduced

### Event Flow
1. Game engine publishes events during turn processing
2. Event bus distributes events synchronously to all subscribers
3. Subscribers can be added for logging, metrics, RL data collection, etc.
4. Action processor publishes events via EventPublisherAdapter

### Usage Example
```go
// Subscribe to events
eventBus.Subscribe(func(event events.Event) {
    switch e := event.(type) {
    case *events.TurnStartedEvent:
        // Handle turn start
    case *events.ActionExecutedEvent:
        // Collect RL experience
    }
})
```

## State Machine Integration

The game now includes a formal state machine to manage game lifecycle:

### Game Phases
1. **PhaseInitializing**: Game object creation
2. **PhaseLobby**: Players joining, configuration
3. **PhaseStarting**: Map generation, player placement
4. **PhaseRunning**: Active gameplay (only phase that accepts actions)
5. **PhasePaused**: Temporary suspension
6. **PhaseEnding**: Winner determination, cleanup
7. **PhaseEnded**: Final state
8. **PhaseError**: Error recovery state
9. **PhaseReset**: Reset without full teardown

### State Transitions
- Initializing → Lobby → Starting → Running
- Running ↔ Paused (pause/resume)
- Running → Ending → Ended
- Most phases → Error (on unrecoverable error)
- Error/Ended → Reset → Initializing

### Engine Integration
- Engine checks state before processing actions
- State transitions publish events
- Pause/Resume methods available on Engine
- Single-player games supported (for RL training)

**Note**: gRPC endpoints still need to be updated to respect state machine

## Development Status

### ✅ Completed
- Core game engine with all mechanics
- gRPC server for multiplayer games
- Fog of war implementation
- Map generation system
- Basic Python client infrastructure
- Docker containerization
- Unit tests for core components
- **Event-driven architecture (Phase 1)**
- **State machine framework (Phase 2 - partial)**

### 🚧 In Progress
- Random agent implementation (Python)
- StreamGame gRPC method for real-time updates
- Multi-agent game orchestration
- Experience collection for RL training
- **State machine gRPC integration**

### 📋 Planned
- Full RL training infrastructure
- Self-play mechanics
- Distributed training on AWS
- Model serving via gRPC
- Tournament/matchmaking system
- Performance optimizations for large-scale training
- **Remaining architecture phases (3-6)**

## Development Notes

### Key Implementation Details

When modifying game mechanics, key files to consider:
- Game balance: `internal/game/constants.go` (reads from config)
- Turn processing: `internal/game/engine.go`
- Movement logic: `internal/game/core/movement.go`
- Visual rendering: `internal/ui/renderer/`
- Fog of war: `internal/game/visibility.go`
- Performance optimization: Check incremental update patterns in `stats.go` and `visibility.go`
- gRPC APIs: `proto/game/v1/game.proto`
- Experience APIs: `proto/experience/v1/experience.proto`
- Python integration: `python/generals_agent/`
- Event handling: `internal/game/events/`
- State management: `internal/game/states/`
- Experience collection: `internal/experience/`

### Important Architectural Decisions

1. **Event System**: Uses synchronous event dispatch for simplicity and determinism. Events include timestamps and game IDs for tracking.

2. **State Machine**: Enforces valid game lifecycle transitions. Single-player games are allowed for RL training scenarios.

3. **Performance**: The codebase uses incremental updates for visibility and stats calculations to optimize performance during RL training.

4. **Testing**: When adding new features, ensure tests handle both single-player and multi-player scenarios.

5. **Logging**: Use structured logging with zerolog. Include relevant context (game ID, turn, player ID) in log entries.

### Common Patterns

- **Incremental Updates**: Used in visibility and stats for performance
- **Reusable Maps**: Engine pre-allocates maps to avoid GC pressure
- **Event Publishing**: All significant game actions publish events
- **State Validation**: State transitions validate preconditions
- **Error Handling**: Use wrapped errors with context for debugging

### TODO/Known Issues

- gRPC server needs to integrate with state machine for proper game lifecycle management
- StreamGame method needs completion for real-time updates
- Python RL agent implementations are in progress
- Full RL training loop integration pending

The project uses:
- **Zerolog** for structured logging throughout the codebase
- **Ebiten v2.8.8** for game graphics and rendering
- **gRPC 1.74.2** for client-server communication
- **Viper** for configuration management
- **Testify** for testing assertions
- **Go 1.23.0** (with toolchain 1.24.0) as the base language version
- **Python 3.8+** for RL agent development