# Python Client for Generals RL

This directory contains the Python client library for interacting with the Generals game server via gRPC.

## Setup

### 1. Activate the Existing Virtual Environment

**Important**: The virtual environment `generalsrl` is already created in the project root. You must activate it before running any Python code:

```bash
# From the project root directory
source generalsrl/bin/activate  # On Linux/Mac
# or
generalsrl\Scripts\activate  # On Windows
```

If the virtual environment doesn't exist, create it first:
```bash
python -m venv generalsrl
```

### 2. Install Python Dependencies

```bash
# From the project root (with virtual environment activated)
make install-python-tools

# Or manually:
pip install -r python/requirements.txt
```

### 3. Generate Python Protocol Buffers

```bash
# From the project root
make generate-python-protos

# Or directly:
./scripts/generate-python-protos.sh
```

### 4. Install the Package (Optional)

For development, you can install the package in editable mode:

```bash
cd python
pip install -e .
```

## Running the Examples

### Start the gRPC Server

First, start the game server in a separate terminal:

```bash
# From the project root
go run cmd/game_server/main.go
```

### Run Random Agent Match

To run a match between two random agents:

```bash
# IMPORTANT: First activate the virtual environment
source generalsrl/bin/activate

# Then run from the project root
python python/scripts/run_random_match.py

# Run multiple games (tournament)
python python/scripts/run_random_match.py --games 10

# Custom board size
python python/scripts/run_random_match.py --width 30 --height 30
```

## Usage

### Basic Example

```python
import grpc
from generals_pb.game.v1 import game_pb2, game_pb2_grpc

# Connect to server
channel = grpc.insecure_channel('localhost:50051')
stub = game_pb2_grpc.GameServiceStub(channel)

# Create a game
create_resp = stub.CreateGame(game_pb2.CreateGameRequest(
    config=game_pb2.GameConfig(width=20, height=20, max_players=2)
))
game_id = create_resp.game_id

# Join the game
join_resp = stub.JoinGame(game_pb2.JoinGameRequest(
    game_id=game_id,
    player_name="PythonBot"
))
player_id = join_resp.player_id
player_token = join_resp.player_token
```

### Creating an Agent

```python
from generals_agent import BaseAgent

class MyAgent(BaseAgent):
    def get_action(self, state):
        # Your agent logic here
        # Use state.action_mask to get valid actions
        valid_actions = self.get_valid_actions_from_mask(state.action_mask)
        return random.choice(valid_actions)
```

## Project Structure

```
python/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── generals_pb/          # Generated protobuf files (after running generation)
│   ├── common/v1/        # Common types
│   └── game/v1/          # Game service definitions
├── generals_agent/       # Agent framework (to be created)
│   ├── base_agent.py     # Base agent class
│   ├── client.py         # gRPC client wrapper
│   └── utils.py          # Helper functions
└── examples/             # Example scripts
    ├── simple_client.py  # Basic client example
    └── play_game.py      # Full game example
```

## Next Steps

1. **Run the game server**: `make run-server` (in another terminal)
2. **Generate Python protos**: `make generate-python-protos`
3. **Run the example**: `python python/examples/simple_client.py`

## Training Agents

For reinforcement learning training:

1. Create multiple game instances
2. Have agents play against each other or themselves
3. Collect experiences (states, actions, rewards)
4. Train your neural network
5. Repeat!

See `examples/training_loop.py` for a complete training example (coming soon).