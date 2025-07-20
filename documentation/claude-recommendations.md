# Claude's Project Recommendations

## Critical Fixes (Blocking Build/Tests)

- [ ] Fix config validation failures by adding sensible defaults when config file is missing
- [ ] Replace panic in map generation with proper error handling
- [ ] Fix mapgen tests that are currently failing

## Security Issues

- [ ] Replace predictable player tokens with cryptographically secure random tokens
- [ ] Add comprehensive input validation to gRPC handlers (player names, board dimensions, coordinates)

## Core Feature Implementation

- [ ] Implement gRPC StreamGame method for real-time game state updates
- [ ] Add proper error context throughout codebase (wrap errors with fmt.Errorf)
- [ ] Implement player elimination tracking (TODO in server.go)

## Reinforcement Learning Infrastructure

- [ ] Create Python gRPC client library for game interaction
- [ ] Implement experience replay buffer for RL training
- [ ] Build self-play training loop infrastructure
- [ ] Create baseline rule-based bots for testing and comparison
- [ ] Add example bot implementations in cmd/grpc_client/examples/

## Production Readiness

- [ ] Set up monitoring with Prometheus metrics (game duration, moves/second, active games)
- [ ] Add structured logging with request IDs for tracing
- [ ] Implement distributed tracing for gRPC calls
- [ ] Add integration tests for multiplayer scenarios
- [ ] Perform load testing with ghz tool for gRPC endpoints

## Performance Optimizations

- [ ] Optimize fog of war calculations with proper caching
- [ ] Implement connection pooling for concurrent games
- [ ] Profile memory usage with many active games
- [ ] Review mutex usage in gameInstance to prevent potential deadlocks

## Development Tooling

- [ ] Integrate golangci-lint for comprehensive code linting
- [ ] Add Makefile or go-task for common development operations
- [ ] Consider using buf for protobuf management
- [ ] Set up pre-commit hooks for code quality checks
- [ ] Use testcontainers-go for better integration testing

## Documentation

- [ ] Create gRPC API documentation with examples
- [ ] Document game mechanics and rules comprehensively
- [ ] Add architecture decision records (ADRs)
- [ ] Create deployment guide for production environments