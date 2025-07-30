# Services Implementation Plan

This document outlines the implementation strategy for the gRPC services architecture that will enable distributed reinforcement learning for the Generals.io game.

## Overview

The services architecture provides a network API layer on top of the existing Go game engine, enabling:
- Distributed game execution across multiple machines
- Python RL agents to interact with the game via gRPC
- Scalable experience collection and model distribution
- Tournament and evaluation infrastructure

## GameService Implementation Plan

### Prerequisites & Missing Components

1. **gRPC Dependencies**
   - [x] Add gRPC and protobuf dependencies to go.mod
   - [x] Install protoc compiler and Go plugins
   - [x] Set up proto generation scripts

2. **Project Structure**
   - [x] Create `proto/` directory for .proto files
   - [x] Create `internal/grpc/` for gRPC server implementations
   - [x] Create `pkg/api/` for generated proto code

### Implementation Steps

#### Phase 1: Proto Definition & Setup

1. Create `proto/game/v1/game.proto` with:
   - [x] Service definition (CreateGame, JoinGame, SubmitAction, GetGameState)
   - [x] Message types (GameState, PlayerState, Action, etc.)
   - [x] Enums (GameStatus, PlayerStatus, ActionError)
   
2. [x] Create `proto/common/v1/common.proto` for shared types

3. Set up proto generation:
   - [x] Create `scripts/generate-protos.sh`
   - [x] Add `//go:generate` directives
   - [x] Configure import paths

#### Phase 2: gRPC Server Implementation

1. Create `internal/grpc/gameserver/server.go`:
   - [x] Implement GameServiceServer interface
   - [x] Wrap existing game.Engine
   - [x] Handle game lifecycle (create, join, play)

2. Implement key methods:
   - [x] `CreateGame`: Initialize new game.Engine instance
   - [x] `JoinGame`: Add players to game
   - [x] `SubmitAction`: Validate and execute moves
   - [x] `GetGameState`: Return current state with fog of war

3. Add game session management:
   - [x] In-memory game registry (map[gameID]*game.Engine)
   - [x] Concurrent access protection (mutexes)
   - [x] Game cleanup/expiration

#### Phase 3: Integration & Adapters

1. Create adapters to convert between:
   - [x] Proto Action ↔ game.Action
   - [x] Proto GameState ↔ game.GameState
   - [x] Handle fog of war in state conversion
   - [x] Legal action mask generation (flattened boolean array)

2. Add authentication/session management:
   - [x] Player ID validation
   - [x] Turn validation (expected_turn_number)
   - [x] Idempotency key handling
   - [x] Player token generation and validation

#### Phase 4: Testing Strategy

1. **Unit Tests**:
   - [x] Test each gRPC method independently
   - [ ] Mock game.Engine for isolated testing
   - [x] Test error cases and edge conditions

2. **Integration Tests**:
   - [ ] Full game flow (create → join → play → finish)
   - [ ] Concurrent player actions
   - [ ] Network error handling

3. **Client Examples**:
   - [ ] Create `examples/grpc-client/` with Go client
   - [ ] Simple bot that plays random moves
   - [ ] Performance testing client

4. **Manual Testing Tools**:
   - [x] gRPC reflection for API exploration
   - [ ] grpcurl scripts for debugging
   - [ ] Simple CLI client for interactive testing

### Additional Considerations

- **Observability**: Add OpenTelemetry tracing and metrics
- **Rate Limiting**: Implement per-player action limits
- **Graceful Shutdown**: Handle ongoing games during server stop
- **Configuration**: Server port, max games, timeouts via config
- **Security**: Validate all inputs, prevent game manipulation

### Estimated Timeline

- Phase 1: ✅ COMPLETED (proto setup)
- Phase 2: ✅ MOSTLY COMPLETED (server implementation) 
- Phase 3: ✅ MOSTLY COMPLETED (adapters)
- Phase 4: 3-4 hours (testing)

## RL Training Integration Tasks

### Phase 5: Python Client Library

1. **Create Python gRPC Client** (`python/generals_client/`):
   - [ ] Generate Python protobuf bindings
   - [ ] Create `GeneralsClient` wrapper class
   - [ ] Implement async/sync game interaction methods
   - [ ] Add connection pooling for multiple games
   - [ ] Create numpy-based state representation converters

2. **OpenAI Gym Environment** (`python/generals_gym/`):
   - [ ] Implement `GeneralsEnv` following gym.Env interface
   - [ ] Define observation space (board state as numpy array)
   - [ ] Define action space (flattened move indices)
   - [ ] Add reward shaping options
   - [ ] Support multiple agent training (self-play)
   - [ ] Add env wrappers for common RL preprocessing

3. **Example RL Agents** (`python/examples/`):
   - [ ] Random agent baseline
   - [ ] Simple DQN implementation
   - [ ] PPO self-play example
   - [ ] A2C distributed training example
   - [ ] Model checkpointing and loading

### Phase 6: Training Infrastructure

1. **Distributed Training Support**:
   - [ ] Create `TrainingCoordinator` service for managing distributed actors
   - [ ] Implement actor-learner architecture pattern
   - [ ] Add experience buffer with prioritized replay
   - [ ] Support for multiple game servers (horizontal scaling)

2. **Monitoring & Metrics**:
   - [ ] Add RL-specific metrics (win rate, episode length, etc.)
   - [ ] TensorBoard integration for training curves
   - [ ] Game replay visualization tools
   - [ ] Performance profiling for bottleneck identification

3. **Model Management**:
   - [ ] Model versioning system
   - [ ] Hot-swapping models during training
   - [ ] Model evaluation pipeline
   - [ ] ELO rating system for model comparison

### Phase 7: Advanced Features

1. **Curriculum Learning**:
   - [ ] Start with smaller boards, increase complexity
   - [ ] Opponent skill progression
   - [ ] Map difficulty progression

2. **League Play**:
   - [ ] Historical model pool for training diversity
   - [ ] Automated tournaments
   - [ ] Matchmaking based on skill level

3. **Human Play Interface**:
   - [ ] Web UI for playing against trained agents
   - [ ] Analysis tools for understanding agent behavior
   - [ ] Training data collection from human games

## Future Services

### ReplayService
- Store game experiences in batches
- Implement priority-based sampling
- S3 integration for long-term storage

### ModelService
- Model versioning and distribution
- PyTorch/TensorFlow serialization support
- Model performance tracking

### TrainingCoordinator
- Manage distributed actors and learners
- Handle failures and restarts
- Dynamic resource allocation

### MetricsService
- Real-time training metrics
- Game statistics aggregation
- Performance monitoring

### ConfigurationService
- Centralized hyperparameter management
- A/B testing support
- Hot configuration updates

## Development Workflow

1. **Local Development**:
   ```bash
   # Start game server
   go run cmd/game_server/main.go
   
   # Test with grpcurl
   grpcurl -plaintext localhost:50051 list
   
   # Run integration tests
   go test ./internal/grpc/...
   ```

2. **Docker Development**:
   ```bash
   # Build server image
   docker build -t generals-grpc-server .
   
   # Run with docker-compose
   docker-compose up game-server
   ```

3. **Proto Changes**:
   ```bash
   # Regenerate proto files
   make generate-protos
   
   # Run compatibility checks
   make proto-breaking-check
   ```

4. **Python Client Development**:
   ```bash
   # Generate Python protobuf files
   python -m grpc_tools.protoc -I./proto --python_out=./python/generals_client/proto --grpc_python_out=./python/generals_client/proto ./proto/game/v1/*.proto
   
   # Install Python client
   cd python && pip install -e .
   
   # Run example agent
   python examples/random_agent.py
   ```

5. **RL Training Workflow**:
   ```bash
   # Start multiple game servers
   docker-compose up --scale game-server=4
   
   # Run distributed training
   python train_ppo.py --num-actors 16 --num-learners 2
   
   # Monitor with TensorBoard
   tensorboard --logdir ./logs
   ```

## Migration Strategy

1. **Backward Compatibility**: Maintain existing game engine API
2. **Gradual Rollout**: Test with simple bots before RL agents
3. **Performance Validation**: Ensure gRPC overhead is acceptable
4. **Monitoring**: Track latency and throughput metrics

## Security Considerations

1. **Authentication**: Player tokens for game access
2. **Authorization**: Validate players can only control their own actions
3. **Rate Limiting**: Prevent action spam
4. **Input Validation**: Sanitize all proto inputs
5. **Network Security**: TLS for production deployments