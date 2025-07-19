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
   - Add gRPC and protobuf dependencies to go.mod
   - Install protoc compiler and Go plugins
   - Set up proto generation scripts

2. **Project Structure**
   - Create `proto/` directory for .proto files
   - Create `internal/grpc/` for gRPC server implementations
   - Create `pkg/api/` for generated proto code

### Implementation Steps

#### Phase 1: Proto Definition & Setup

1. Create `proto/game/v1/game.proto` with:
   - Service definition (CreateGame, JoinGame, SubmitAction, GetGameState, StreamGame)
   - Message types (GameState, PlayerState, Action, etc.)
   - Enums (GameStatus, PlayerStatus, ActionError)
   
2. Create `proto/common/v1/common.proto` for shared types

3. Set up proto generation:
   - Create `scripts/generate-protos.sh`
   - Add `//go:generate` directives
   - Configure import paths

#### Phase 2: gRPC Server Implementation

1. Create `internal/grpc/gameserver/server.go`:
   - Implement GameServiceServer interface
   - Wrap existing game.Engine
   - Handle game lifecycle (create, join, play)

2. Implement key methods:
   - `CreateGame`: Initialize new game.Engine instance
   - `JoinGame`: Add players to game
   - `SubmitAction`: Validate and execute moves
   - `GetGameState`: Return current state with fog of war
   - `StreamGame`: Server-side streaming of game updates

3. Add game session management:
   - In-memory game registry (map[gameID]*game.Engine)
   - Concurrent access protection (mutexes)
   - Game cleanup/expiration

#### Phase 3: Integration & Adapters

1. Create adapters to convert between:
   - Proto Action ↔ game.Action
   - Proto GameState ↔ game.GameState
   - Handle fog of war in state conversion

2. Add authentication/session management:
   - Player ID validation
   - Turn validation (expected_turn_number)
   - Idempotency key handling

#### Phase 4: Testing Strategy

1. **Unit Tests**:
   - Test each gRPC method independently
   - Mock game.Engine for isolated testing
   - Test error cases and edge conditions

2. **Integration Tests**:
   - Full game flow (create → join → play → finish)
   - Concurrent player actions
   - Network error handling

3. **Client Examples**:
   - Create `examples/grpc-client/` with Go client
   - Simple bot that plays random moves
   - Performance testing client

4. **Manual Testing Tools**:
   - gRPC reflection for API exploration
   - grpcurl scripts for debugging
   - Simple CLI client for interactive testing

### Additional Considerations

- **Observability**: Add OpenTelemetry tracing and metrics
- **Rate Limiting**: Implement per-player action limits
- **Graceful Shutdown**: Handle ongoing games during server stop
- **Configuration**: Server port, max games, timeouts via config
- **Security**: Validate all inputs, prevent game manipulation

### Estimated Timeline

- Phase 1: 2-3 hours (proto setup)
- Phase 2: 4-6 hours (server implementation)
- Phase 3: 2-3 hours (adapters)
- Phase 4: 3-4 hours (testing)

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
   go run cmd/grpc_server/main.go
   
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