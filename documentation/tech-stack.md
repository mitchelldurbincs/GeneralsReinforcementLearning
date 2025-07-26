# Technology Stack

This document outlines the technologies used in the Generals Reinforcement Learning project and the rationale behind each choice.

## Core Technologies

### Programming Language
- **Go 1.24.0**: Primary language for the entire application
  - **Why**: High performance for game logic, excellent concurrency support for multiplayer gaming, and efficient memory management crucial for RL training workloads

### Game Development
- **Ebiten v2.8.8**: 2D game engine for Go
  - **Why**: Native Go integration, cross-platform support, and efficient rendering for the game's grid-based visuals
  - **Used for**: UI client rendering, board visualization, and input handling

### Communication & APIs
- **gRPC & Protocol Buffers**: RPC framework and serialization
  - **Why**: Efficient binary serialization for game state updates, strongly-typed API contracts, and excellent performance for real-time multiplayer communication
  - **Used for**: Client-server communication, game state synchronization, and future RL agent communication

### Logging & Observability
- **Zerolog v1.34.0**: Structured logging library
  - **Why**: Zero-allocation JSON logging for minimal performance impact during gameplay, structured logs for better debugging of complex game states
  - **Used for**: Application-wide logging, debugging, and monitoring

### Testing
- **Testify v1.10.0**: Testing toolkit with assertions
  - **Why**: Rich assertion library that makes tests more readable and maintainable
  - **Used for**: Unit tests across all packages

## Infrastructure & Deployment

### Containerization
- **Docker**: Container platform
  - **Why**: Consistent deployment across environments, easy scaling for multiple game servers
  - **Multi-stage builds**: Using golang:1.24-alpine for building and alpine:3.21 for runtime to minimize image size
  - **Used for**: Packaging the game server for deployment

### Infrastructure as Code
- **Terraform**: Infrastructure provisioning
  - **Why**: Declarative infrastructure management, version-controlled infrastructure changes
  - **Used for**: AWS resource provisioning including EC2, ECR, VPC, and S3

### Cloud Platform
- **AWS**: Cloud infrastructure provider
  - **EC2**: For running game servers and RL training instances
  - **ECR**: Container registry for Docker images
  - **S3**: Storage for game replays and RL model checkpoints
  - **VPC**: Network isolation and security
  - **CloudWatch**: Monitoring and logging

## Development Tools

### Protocol Buffer Tools
- **protoc**: Protocol buffer compiler
  - **protoc-gen-go-grpc**: Go code generation for gRPC
  - **Why**: Automated code generation from proto definitions ensures type safety and reduces boilerplate

### Build Tools
- **Go Modules**: Dependency management
  - **Why**: Native Go dependency management with version pinning

## Architecture Decisions

### Headless Server Design
- Separate game server (`cmd/game_server`) from UI client (`cmd/ui_client`)
- **Why**: Enables running game simulations without graphics overhead for RL training, better separation of concerns

### Binary Protocol (gRPC/Protobuf)
- Chose binary serialization over JSON/REST
- **Why**: Lower latency for real-time game updates, smaller payload sizes for game state synchronization

### Incremental State Updates
- Fog of war and visibility calculations use incremental updates
- **Why**: Performance optimization for large game boards, reduces computational overhead

### Terminal-Based Rendering Option
- ANSI color codes for terminal visualization
- **Why**: Debugging and development without GUI dependencies, useful for server-side testing

## Future Considerations

### Reinforcement Learning Stack
- The infrastructure is designed to support future RL integration:
  - Headless game server for fast simulation
  - gRPC for efficient agent-server communication
  - S3 for model checkpoint storage
  - Separate EC2 instances for training workloads

### Scalability
- Docker and Terraform setup enables horizontal scaling
- gRPC supports load balancing for multiple game servers
- Stateless server design allows for easy scaling