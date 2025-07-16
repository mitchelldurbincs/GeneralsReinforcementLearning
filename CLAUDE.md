# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Go-based implementation of a Generals.io-style strategic territory control game with reinforcement learning capabilities. The game features army movement, territory capture, fog of war, and multiplayer support.

## Development Commands

### Building
```bash
# Build the game server (headless)
go build -o game_server ./cmd/game_server

# Build the UI client (with graphics)
go build -o ui_client ./cmd/ui_client

# Build for Docker deployment
CGO_ENABLED=0 go build -o generals ./cmd/game
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
- **engine.go**: Main game loop, turn processing, win condition checking
- **state.go**: Game state management, player tracking
- **constants.go**: Game balance parameters (production rates, growth intervals)
- **core/**: Low-level game mechanics
  - `board.go`: Grid-based game board
  - `movement.go`: Army movement and combat resolution
  - `player.go`: Player state and actions
  - `tile.go`: Tile types (empty, mountain, city, general)
- **mapgen/**: Procedural map generation with city and mountain placement

### UI System (`internal/ui/`)
- **renderer/**: Ebiten-based rendering for board, units, and fog of war
- **input/**: Mouse and keyboard handling for player interactions

### Entry Points (`cmd/`)
- **game_server/**: Headless server for game logic (planned for RL training)
- **ui_client/**: Graphical client for human players

### Key Game Parameters
- City spawn ratio: 1 per 20 tiles (~5% of map)
- City starting army: 40 units
- Production: 1 army/turn for generals and cities
- Normal tile growth: Every 25 turns
- Fog of war: Recently implemented, hides enemy positions

### Deployment
- Docker multi-stage builds for containerization
- Terraform configuration for AWS deployment (EC2, ECR, S3, VPC)

## Development Notes

When modifying game mechanics, key files to consider:
- Game balance: `internal/game/constants.go`
- Turn processing: `internal/game/engine.go`
- Movement logic: `internal/game/core/movement.go`
- Visual rendering: `internal/ui/renderer/`

The project uses Zerolog for structured logging throughout the codebase.