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

# Manage dependencies
go mod tidy
go mod download
```

## Architecture Overview

### Core Game Logic (`internal/game/`)
- **engine.go**: Main game loop, turn processing, action execution, win condition checking
- **state.go**: Game state management, player and board state tracking
- **constants.go**: Game balance parameters (production rates, growth intervals)
- **visibility.go**: Fog of war implementation with incremental updates (3x3 visibility around owned tiles)
- **stats.go**: Player statistics tracking with optimized incremental updates
- **rendering.go**: Terminal-based board visualization with ANSI colors
- **demo_helpers.go**: Random action generation for testing and demonstrations
- **core/**: Low-level game mechanics
  - `board.go`: 2D grid-based game board implementation
  - `movement.go`: Army movement and combat resolution
  - `player.go`: Player state and properties
  - `tile.go`: Tile types (empty, mountain, city, general)
  - `action.go`: Action types and validation
  - `utils.go`: Core utility functions
- **mapgen/**: Procedural map generation
  - `generator.go`: Map generation with city and mountain placement

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

### Networking (`internal/grpc/`)
- **gameserver/**: gRPC server implementation
  - `server.go`: Main server logic, game management, player sessions
  - `stream_manager.go`: Real-time game state streaming (in development)
- **proto/**: Protocol buffer definitions for client-server communication
- **client/**: Go client library for connecting to the game server

### Python Integration (`python/`)
- **grpc_client.py**: Python gRPC client for RL agents
- **random_agent.py**: Basic random agent implementation (in progress)
- **Environment classes planned**: OpenAI Gym-compatible wrappers

### Key Game Parameters
- City spawn ratio: 1 per 20 tiles (~5% of map)
- City starting army: 40 units
- Production: 1 army/turn for generals and cities
- Normal tile growth: Every 25 turns
- Fog of war: Fully implemented with toggle via `FogOfWarEnabled` flag
- Minimum general spacing: 5 (Manhattan distance)
- Turn time limit: 500ms default (configurable)
- Map sizes: Small (10x10), Medium (15x15), Large (20x20)
- Maximum players: 8 (limited by distinct colors)

### Deployment
- Docker multi-stage builds for containerization (golang:1.24-alpine → alpine:3.21)
- Terraform configuration for AWS deployment:
  - Individual .tf files for each component (compute, ECR, networking, security, etc.)
  - EC2 instances for game server and RL trainer
  - ECR for container registry
  - S3 for storage
  - VPC and security groups

## Development Status

### ✅ Completed
- Core game engine with all mechanics
- gRPC server for multiplayer games
- Fog of war implementation
- Map generation system
- Basic Python client infrastructure
- Docker containerization
- Unit tests for core components

### 🚧 In Progress
- Random agent implementation (Python)
- StreamGame gRPC method for real-time updates
- Multi-agent game orchestration
- Experience collection for RL training

### 📋 Planned
- Full RL training infrastructure
- Self-play mechanics
- Distributed training on AWS
- Model serving via gRPC
- Tournament/matchmaking system
- Performance optimizations for large-scale training

## Development Notes

When modifying game mechanics, key files to consider:
- Game balance: `internal/game/constants.go`
- Turn processing: `internal/game/engine.go`
- Movement logic: `internal/game/core/movement.go`
- Visual rendering: `internal/ui/renderer/`
- Fog of war: `internal/game/visibility.go`
- Performance optimization: Check incremental update patterns in `stats.go` and `visibility.go`
- gRPC APIs: `internal/grpc/proto/game.proto`
- Python integration: `python/` directory

The project uses:
- **Zerolog** for structured logging throughout the codebase
- **Ebiten v2.8.8** for game graphics and rendering
- **gRPC 1.70.0** for client-server communication
- **Viper** for configuration management
- **Testify** for testing assertions
- **Go 1.24.0** as the base language version
- **Python 3.8+** for RL agent development