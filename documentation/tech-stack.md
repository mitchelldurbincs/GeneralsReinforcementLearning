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
- **gRPC v1.73.0 & Protocol Buffers v1.36.6**: RPC framework and serialization
  - **Why**: Efficient binary serialization for game state updates, strongly-typed API contracts, and excellent performance for real-time multiplayer communication
  - **Used for**: Client-server communication, game state synchronization, and RL agent communication
  - **Code Generation**: Supports both Go and Python client generation from same proto definitions
  - **Server Implementation**: Dedicated gRPC server at `cmd/grpc_server/` for network-based gameplay

### Configuration Management
- **Viper v1.20.1**: Configuration management library
  - **Why**: Support for multiple configuration formats, environment variables, and hot-reloading
  - **Used for**: Managing game settings, server configuration, and environment-specific values
- **fsnotify v1.9.0**: File system notifications
  - **Why**: Enables configuration hot-reloading when config files change
  - **Used for**: Watching configuration files for changes during development

### Image Processing
- **golang.org/x/image v0.27.0**: Extended image processing
  - **Why**: Additional image formats and processing utilities beyond standard library
  - **Used for**: Enhanced graphics rendering in the UI client

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
  - **grpcio-tools**: Python code generation for gRPC
  - **Why**: Automated code generation from proto definitions ensures type safety and reduces boilerplate
  - **Scripts**: `generate-protos.sh` for Go, `generate-python-protos.sh` for Python

### Build Tools
- **Go Modules**: Dependency management
  - **Why**: Native Go dependency management with version pinning

## Python Client Stack

### Language & Runtime
- **Python 3.12**: Client implementation language
  - **Why**: Popular for RL research, extensive ML libraries, easy prototyping
  - **Used for**: Building RL agents, research experiments, and testing

### Python Dependencies
- **grpcio >= 1.48.0**: gRPC runtime for Python
  - **Why**: Native Python support for gRPC communication with game server
  - **Used for**: Client-server communication
- **grpcio-tools >= 1.48.0**: Protocol buffer compiler for Python
  - **Why**: Generate Python stubs from proto definitions
  - **Used for**: Code generation from proto files
- **protobuf >= 3.20.0**: Protocol buffer runtime
  - **Why**: Serialization/deserialization of game messages
  - **Used for**: Message handling between client and server
- **numpy >= 1.21.0**: Numerical computing library
  - **Why**: Efficient array operations for game state representation
  - **Used for**: RL agent state processing and calculations

### Python Client Architecture
- Located in `python/` directory
- Generated protobuf stubs in `generals_pb/`
- Example clients in `python/examples/`
- Simple client implementation for testing and RL agent development

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

### Configuration Management Strategy
- YAML-based configuration files with Viper
- **Configuration Files**:
  - `config.yaml`: Base configuration with defaults
  - `config.dev.yaml`: Development environment overrides
  - `config.prod.yaml`: Production environment settings
- **Why**: Flexible configuration management, environment-specific settings, hot-reloading support
- **Features**: Hierarchical configuration, environment variable overrides, type safety

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