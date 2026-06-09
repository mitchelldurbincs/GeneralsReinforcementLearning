# Generals Reinforcement Learning

A high-performance [Generals.io](https://generals.io)-style territory control game built in Go, designed as a reinforcement learning training platform. The game engine supports distributed self-play, experience collection, and Python-based agent development via gRPC.

> **Not a public multiplayer server.** This is an RL research platform: bots play each other at scale, experiences stream to training pipelines, and humans drop in occasionally to evaluate agent quality.

---

## What It Is

- **Turn-based strategy game**: Players move armies to capture territory and eliminate opponents
- **RL training environment**: The game server streams (state, action, reward, next\_state, done) tuples to Python training pipelines in real time
- **Distributed-ready**: A single server handles 100+ concurrent games; experiences batch and stream over gRPC at 50,000+ tuples/second
- **Human-playable**: An Ebiten-based graphical client lets humans play against bots for evaluation

### Game Mechanics

| Mechanic | Details |
|---|---|
| Map | 2D grid with empty tiles, mountains (impassable), cities, and generals |
| Armies | Move to adjacent tiles; larger army wins combat, difference remains |
| Production | Generals and cities produce 1 army/turn; normal tiles grow every 25 turns |
| Fog of war | 3×3 visibility radius around owned tiles (configurable) |
| Victory | Capture all enemy generals, or control the entire map |
| Map sizes | Small 10×10, Medium 15×15, Large 20×20 (configurable) |
| Players | 2–4 per game |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Game Server (Go)                        │
│                                                             │
│  Engine → EventBus → ExperienceCollector → StreamAggregator │
│                ↓                                   ↓        │
│           StateMachine                      BatchProcessor  │
│                                                   ↓        │
│                                          gRPC Stream (:50051)│
└──────────────────────────────┬──────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
   ┌──────────▼──────────┐          ┌──────────▼──────────┐
   │   UI Client (Go)    │          │  Python RL Agents   │
   │  Ebiten graphics    │          │  gRPC client + DQN  │
   └─────────────────────┘          └─────────────────────┘
```

### Key Packages

| Path | Role |
|---|---|
| `internal/game/` | Core engine: turn processing, combat, production, fog of war |
| `internal/game/core/` | Board, movement, actions, coordinates |
| `internal/game/events/` | Synchronous event bus (GameStarted, TurnEnded, CombatResolved, …) |
| `internal/game/states/` | State machine: Initializing → Lobby → Running ⇄ Paused → Ended |
| `internal/game/mapgen/` | Procedural map generation |
| `internal/grpc/gameserver/` | gRPC server: game lifecycle, streaming, experience aggregation |
| `internal/experience/` | Experience buffers, collectors, batch processors |
| `internal/ui/renderer/` | Ebiten rendering (board, fog, units) |
| `proto/` | Protobuf v3 API definitions for game and experience services |
| `python/generals_agent/` | Python base agent class, gRPC client, session management |

---

## Getting Started

### Prerequisites

- Go 1.24+
- Python 3.8+ (for RL agents)
- `protoc` + Go/Python plugins (only needed to regenerate protos)

### Build

```bash
# Headless game server
go build -o game_server ./cmd/game_server

# Graphical UI client
go build -o ui_client ./cmd/ui_client
```

### Run

```bash
# Start the game server (default port 50051)
go run cmd/game_server/main.go

# Optional flags
go run cmd/game_server/main.go -port 50051 -max-games 100 -log-level info

# Start the graphical client (human vs bots)
go run cmd/ui_client/main.go

# All-bot mode with 3 players on a 20×15 map
go run cmd/ui_client/main.go -players 3 -human -1 -width 20 -height 15
```

### Python Agents

```bash
# Activate the virtual environment (required)
source generalsrl/bin/activate

# Run a random-agent match
python python/scripts/run_random_match.py

# Test the gRPC connection
python python/test_grpc_client.py
```

### Tests

```bash
go test ./...
go test -cover ./...

# Individual packages
go test ./internal/game/core
go test ./internal/game/mapgen
go test ./internal/ui/renderer
```

---

## Configuration

Configuration lives in `config/` and is managed by Viper. Values cascade: defaults → `config.yaml` → environment-specific override → CLI flags.

```yaml
# config/config.yaml (abbreviated)
game:
  fog_of_war:
    enabled: true
  city_spawn_ratio: 20      # 1 city per N tiles
  general_min_spacing: 5    # Manhattan distance
  normal_tile_growth_interval: 25

server:
  grpc_address: "0.0.0.0:50051"
  max_games: 100

ui:
  window_width: 800
  window_height: 600
  tile_size: 32
```

Environment-specific files: `config.dev.yaml`, `config.prod.yaml`, `config/rlconfig.yaml`.

---

## Experience Streaming

The platform streams RL training tuples from running games to Python clients in real time.

```
Game Engine
  → ExperienceCollector   (captures state/action/reward/next_state/done)
  → Buffer                (10k capacity per game, thread-safe)
  → StreamAggregator      (merges streams from multiple games)
  → BatchProcessor        (batches of 32, 100ms timeout)
  → gRPC Stream           (to Python training client)
```

**Performance targets**: 50,000+ experiences/sec, 1,000+ concurrent games, <100ms p95 batch latency.

### Python Client

```python
from experience_stream_client import ExperienceStreamClient, ExperienceConfig

client = ExperienceStreamClient(ExperienceConfig(
    server_address="localhost:50051",
    batch_size=32,
    follow=True,
))
client.connect()
client.start_streaming()

batch = client.get_batch(32)  # ready for training step
```

---

## Writing an Agent

Subclass `BaseAgent` in Python:

```python
from generals_agent.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def select_action(self, game_state):
        # game_state is a game_pb2.GameState protobuf
        # return a game_pb2.Action
        ...

    def on_game_start(self, initial_state):
        # optional setup
        pass
```

---

## Deployment

### Docker

```bash
# Build image
docker build -f deploy/Dockerfile -t generals-rl .

# Run server
docker run -p 50051:50051 generals-rl
```

### AWS (Terraform)

Infrastructure definitions live in `deploy/terraform/`:
- EC2 instances for game server and RL trainer
- ECR for container images
- S3 for experience storage and model checkpoints
- VPC, security groups, IAM roles, CloudWatch monitoring

---

## Development Status

| Area | Status |
|---|---|
| Core game engine | Done |
| gRPC server | Done |
| Fog of war | Done |
| Map generation | Done |
| Event system | Done |
| State machine | Done |
| Experience collection & streaming | Done |
| Docker / Terraform | Done |
| Python base agent + gRPC client | Done |
| Python RL training pipeline | In progress |
| Self-play orchestration | Planned |
| OpenAI Gym wrapper | Planned |
| Distributed training (AWS) | Planned |
| Model serving & tournaments | Planned |

---

## Project Structure

```
GeneralsReinforcementLearning/
├── cmd/
│   ├── game_server/        # Headless gRPC game server
│   └── ui_client/          # Ebiten graphical client
├── config/                 # YAML configuration files
├── deploy/
│   ├── Dockerfile          # Multi-stage production build
│   └── terraform/          # AWS infrastructure
├── docker/                 # Additional Docker configs
├── internal/
│   ├── common/             # Shared utilities
│   ├── config/             # Config loading
│   ├── experience/         # Experience collection & buffers
│   ├── game/               # Game engine + subsystems
│   ├── grpc/               # gRPC server implementation
│   ├── monitoring/         # Metrics
│   └── ui/                 # Renderer + input handling
├── proto/                  # Protobuf definitions
│   ├── common/v1/
│   ├── experience/v1/
│   └── game/v1/
├── python/
│   ├── generals_agent/     # Agent base class & gRPC client
│   ├── generals_pb/        # Generated protobuf bindings
│   ├── examples/
│   └── scripts/
├── scripts/
│   └── generate-protos.sh  # Regenerate Go + Python protos
├── go.mod
└── go.sum
```

---

## License

See [LICENSE](LICENSE) if present, or contact the repository owner.
