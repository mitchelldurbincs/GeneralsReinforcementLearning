# Todo List

## High Priority (Do Next)
- [ ] Complete gRPC server implementation
  - [ ] Implement SubmitAction method in gameserver/server.go
  - [ ] Implement StreamGame method for real-time updates
  - [ ] Create cmd/grpc_server/main.go to actually serve the gRPC API
- [ ] Add gRPC client implementation
  - [ ] Create Go client library in pkg/client/
  - [ ] Add example client in cmd/grpc_client/
  - [ ] Create integration tests using the client

## Medium Priority
- [ ] Create Reinforcement Learning agent integration
  - [ ] Python gRPC client for RL agents
  - [ ] Environment wrapper following OpenAI Gym interface
  - [ ] Basic self-play training loop
- [ ] Add comprehensive tests
  - [ ] Unit tests for gRPC service methods
  - [ ] Integration tests for full game flow
  - [ ] Load/stress testing for multiplayer scenarios
- [ ] Docker and deployment improvements
  - [ ] Update Dockerfile to run gRPC server
  - [ ] Docker Compose for local development
  - [ ] Health checks and monitoring endpoints

## Low Priority
- [ ] Enhanced game features
  - [ ] Implement SPLIT and DEFEND action types
  - [ ] Add game history/move tracking
  - [ ] Leaderboard and statistics service
- [ ] UI improvements
  - [ ] Connect UI client to gRPC server
  - [ ] Add multiplayer lobby interface
  - [ ] Spectator mode via StreamGame
- [ ] Add replay functionality (kept from original)
  - [ ] Record games to file
  - [ ] Replay viewer in UI
  - [ ] Export replays for analysis

## Completed
- [x] Create proto definitions
- [x] Generate protobuf/gRPC code
- [x] Basic gRPC server structure
- [x] CreateGame implementation
- [x] JoinGame implementation
- [x] GetGameState implementation
- [x] Fog of war (already implemented)
- [x] Ebiten engine (UI client exists)
- [x] Non-rendering mode (headless server exists)
- [x] Create Dockerfile (basic version exists)