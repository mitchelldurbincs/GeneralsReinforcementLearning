# Todo List

## High Priority (Do Next)
- [ ] Complete StreamGame implementation in gRPC server
  - [ ] Implement server-side streaming for real-time game updates
  - [ ] Add event types for different game state changes
  - [ ] Test with multiple concurrent streams
- [ ] Create gRPC client examples
  - [ ] Complete the Go client in examples/grpc-client/
  - [ ] Add Python client example for RL integration
  - [ ] Create simple bot that plays via gRPC
- [ ] Implement Reinforcement Learning infrastructure
  - [ ] Create Python environment wrapper (OpenAI Gym compatible)
  - [ ] Implement experience replay buffer
  - [ ] Add basic self-play training loop
  - [ ] Create model serving infrastructure

## Medium Priority
- [ ] Add comprehensive testing
  - [ ] Integration tests for full multiplayer game flow
  - [ ] Load/stress testing for concurrent games
  - [ ] Test turn timeout handling
  - [ ] Test player disconnection scenarios
- [ ] Implement additional gRPC services
  - [ ] ReplayService for experience collection
  - [ ] ModelService for policy distribution
  - [ ] MatchMakerService for game coordination
- [ ] Docker and deployment improvements
  - [ ] Complete Terraform infrastructure setup
  - [ ] Add Kubernetes manifests for container orchestration
  - [ ] Implement health checks and monitoring
  - [ ] Add Prometheus metrics endpoints

## Low Priority
- [ ] Enhanced game features
  - [ ] Implement SPLIT and DEFEND action types (currently only MOVE)
  - [ ] Add surrender functionality
  - [ ] Implement spectator mode
  - [ ] Add game history/replay recording
- [ ] UI improvements
  - [ ] Connect UI client to remote gRPC server
  - [ ] Add multiplayer lobby interface
  - [ ] Implement reconnection logic
  - [ ] Add chat functionality
- [ ] Performance optimizations
  - [ ] Optimize fog of war calculations
  - [ ] Add game state caching
  - [ ] Implement connection pooling
  - [ ] Profile and optimize hot paths

## Completed
- [x] Create proto definitions (game.proto, common.proto)
- [x] Generate protobuf/gRPC code with proper build system
- [x] Implement gRPC server with full game lifecycle
- [x] CreateGame implementation with configuration
- [x] JoinGame implementation with authentication
- [x] SubmitAction implementation with turn-based logic
- [x] GetGameState implementation with fog of war
- [x] Add Viper configuration management
- [x] Implement fog of war system
- [x] Create Ebiten-based UI client
- [x] Implement headless game server
- [x] Create basic Dockerfile
- [x] Add comprehensive unit tests for gRPC server
- [x] Implement turn timers and timeouts
- [x] Add graceful shutdown handling
- [x] Create proto generation scripts
- [x] Add health check service
- [x] Implement gRPC reflection for debugging

## Notes
- StreamGame method exists but returns Unimplemented - this is the next major feature to implement
- The RL-specific features (experience replay, model serving) are not yet implemented
- Current implementation focuses on core multiplayer gameplay
- All game mechanics are working: movement, combat, fog of war, win conditions