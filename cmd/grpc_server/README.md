# gRPC Game Server

This is the gRPC server implementation for the Generals.io-style game, providing a network API for game clients and reinforcement learning agents.

## Usage

### Running the Server

```bash
# Run with default settings
go run cmd/grpc_server/main.go

# Run with custom configuration
go run cmd/grpc_server/main.go \
  --port=50051 \
  --host=0.0.0.0 \
  --log-level=debug \
  --turn-timeout=5000 \
  --max-games=50
```

### Command-line Flags

- `--port` (default: 50051) - The gRPC server port
- `--host` (default: 0.0.0.0) - The server bind address
- `--log-level` (default: info) - Log level: debug, info, warn, error
- `--turn-timeout` (default: 0) - Default turn timeout in milliseconds (0 = no timeout)
- `--max-games` (default: 100) - Maximum concurrent games
- `--enable-reflection` (default: true) - Enable gRPC reflection for debugging

### Environment Variables

- `APP_ENV=production` - Switches to JSON logging output (default: pretty console output)

## Features

### Core Services

1. **GameService** - Main game API
   - `CreateGame` - Create a new game instance
   - `JoinGame` - Join an existing game
   - `SubmitAction` - Submit player actions
   - `GetGameState` - Get current game state
   - `StreamGame` - Stream real-time game updates (not yet implemented)

2. **Health Service** - Kubernetes-compatible health checks
   - Standard gRPC health check protocol
   - Liveness and readiness probes

### Middleware

- **Logging** - Structured logging for all RPC calls
- **Recovery** - Panic recovery with proper error responses
- **Reflection** - gRPC reflection for debugging with tools like `grpcurl`

### Production Features

- Graceful shutdown with 5-second grace period
- Structured JSON logging in production mode
- Health checks for load balancer integration
- Signal handling (SIGINT, SIGTERM)

## Testing the Server

### Using grpcurl

```bash
# List available services
grpcurl -plaintext localhost:50051 list

# Describe the GameService
grpcurl -plaintext localhost:50051 describe generals.game.v1.GameService

# Create a new game
grpcurl -plaintext -d '{
  "config": {
    "width": 20,
    "height": 20,
    "max_players": 2,
    "fog_of_war": true,
    "turn_time_ms": 5000
  }
}' localhost:50051 generals.game.v1.GameService/CreateGame

# Check health
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

### Using a Go Client

```go
package main

import (
    "context"
    "log"
    "google.golang.org/grpc"
    gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

func main() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()

    client := gamev1.NewGameServiceClient(conn)
    
    // Create a game
    resp, err := client.CreateGame(context.Background(), &gamev1.CreateGameRequest{
        Config: &gamev1.GameConfig{
            Width: 20,
            Height: 20,
            MaxPlayers: 2,
            FogOfWar: true,
        },
    })
    if err != nil {
        log.Fatal(err)
    }
    
    log.Printf("Created game: %s", resp.GameId)
}
```

## Docker Support

Build and run with Docker:

```bash
# Build the image
docker build -t generals-grpc-server -f deploy/Dockerfile --target grpc-server .

# Run the container
docker run -p 50051:50051 generals-grpc-server
```

## Future Enhancements

The following features are planned but not yet implemented:

- Prometheus metrics endpoint for monitoring
- OpenTelemetry tracing integration
- Rate limiting per player
- Authentication middleware
- Periodic cleanup of abandoned games
- Configuration hot-reload support
- StreamGame method implementation
- WebSocket gateway for browser clients