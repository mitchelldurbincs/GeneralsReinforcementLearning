#!/bin/bash

# Script to generate Python code from proto files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure we're in the project root
cd "$(dirname "$0")/.."

echo -e "${YELLOW}Starting Python proto generation...${NC}"

# Check if generalsrl venv exists and use it
if [ -d "generalsrl" ] && [ -f "generalsrl/bin/python" ]; then
    echo "Using generalsrl virtual environment..."
    PYTHON="./generalsrl/bin/python"
    PIP="./generalsrl/bin/pip"
else
    echo "Using system Python (no generalsrl venv found)..."
    PYTHON="python3"
    PIP="pip3"
fi

# Check if grpcio-tools is installed
if ! $PYTHON -c "import grpc_tools" 2>/dev/null; then
    echo -e "${RED}Error: grpcio-tools not installed${NC}"
    echo "Please install it with: $PIP install grpcio-tools"
    echo "Or run: make install-python-tools"
    exit 1
fi

# Create output directories if they don't exist
echo "Creating Python package directories..."
mkdir -p python/generals_pb/common/v1
mkdir -p python/generals_pb/game/v1
mkdir -p python/generals_pb/experience/v1

# Create __init__.py files for proper Python packages
touch python/__init__.py
touch python/generals_pb/__init__.py
touch python/generals_pb/common/__init__.py
touch python/generals_pb/common/v1/__init__.py
touch python/generals_pb/game/__init__.py
touch python/generals_pb/game/v1/__init__.py
touch python/generals_pb/experience/__init__.py
touch python/generals_pb/experience/v1/__init__.py

# Generate common proto files
echo -e "${GREEN}Generating common proto files...${NC}"
$PYTHON -m grpc_tools.protoc \
    -I proto \
    --python_out=python/generals_pb \
    --grpc_python_out=python/generals_pb \
    proto/common/v1/common.proto

# Generate game proto files
echo -e "${GREEN}Generating game proto files...${NC}"
$PYTHON -m grpc_tools.protoc \
    -I proto \
    --python_out=python/generals_pb \
    --grpc_python_out=python/generals_pb \
    proto/game/v1/game.proto

# Generate experience proto files
echo -e "${GREEN}Generating experience proto files...${NC}"
$PYTHON -m grpc_tools.protoc \
    -I proto \
    --python_out=python/generals_pb \
    --grpc_python_out=python/generals_pb \
    proto/experience/v1/experience.proto

# Fix imports in generated files (protoc generates absolute imports that need adjustment)
echo "Fixing Python imports..."

# Fix imports in generated files
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    # Fix imports in game_pb2.py
    sed -i '' 's/^from common\.v1 import/from generals_pb.common.v1 import/' python/generals_pb/game/v1/game_pb2.py
    # Fix imports in game_pb2_grpc.py  
    sed -i '' 's/^from common\.v1 import/from generals_pb.common.v1 import/' python/generals_pb/game/v1/game_pb2_grpc.py 2>/dev/null || true
    sed -i '' 's/^from game\.v1 import/from generals_pb.game.v1 import/' python/generals_pb/game/v1/game_pb2_grpc.py 2>/dev/null || true
    # Fix imports in experience_pb2.py
    sed -i '' 's/^from common\.v1 import/from generals_pb.common.v1 import/' python/generals_pb/experience/v1/experience_pb2.py
    # Fix imports in experience_pb2_grpc.py
    sed -i '' 's/^from common\.v1 import/from generals_pb.common.v1 import/' python/generals_pb/experience/v1/experience_pb2_grpc.py 2>/dev/null || true
    sed -i '' 's/^from experience\.v1 import/from generals_pb.experience.v1 import/' python/generals_pb/experience/v1/experience_pb2_grpc.py 2>/dev/null || true
else
    # Linux
    # Fix imports in game_pb2.py
    sed -i 's/^from common\.v1 import/from generals_pb.common.v1 import/' python/generals_pb/game/v1/game_pb2.py
    # Fix imports in game_pb2_grpc.py
    sed -i 's/^from common\.v1 import/from generals_pb.common.v1 import/' python/generals_pb/game/v1/game_pb2_grpc.py 2>/dev/null || true
    sed -i 's/^from game\.v1 import/from generals_pb.game.v1 import/' python/generals_pb/game/v1/game_pb2_grpc.py 2>/dev/null || true
    # Fix imports in experience_pb2.py
    sed -i 's/^from common\.v1 import/from generals_pb.common.v1 import/' python/generals_pb/experience/v1/experience_pb2.py
    # Fix imports in experience_pb2_grpc.py
    sed -i 's/^from common\.v1 import/from generals_pb.common.v1 import/' python/generals_pb/experience/v1/experience_pb2_grpc.py 2>/dev/null || true
    sed -i 's/^from experience\.v1 import/from generals_pb.experience.v1 import/' python/generals_pb/experience/v1/experience_pb2_grpc.py 2>/dev/null || true
fi

# Create a setup.py for the Python package
cat > python/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="generals-pb",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "grpcio>=1.48.0",
        "protobuf>=3.20.0",
    ],
    python_requires=">=3.7",
    description="Python gRPC client for Generals Reinforcement Learning",
    author="Your Name",
    license="MIT",
)
EOF

# Create requirements.txt for Python dependencies
cat > python/requirements.txt << 'EOF'
grpcio>=1.48.0
grpcio-tools>=1.48.0
protobuf>=3.20.0
numpy>=1.21.0
EOF

# Create a simple example client
mkdir -p python/examples
cat > python/examples/simple_client.py << 'EOF'
#!/usr/bin/env python3
"""
Simple example client for the Generals game server.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import grpc
from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2

def create_game(stub):
    """Create a new game."""
    request = game_pb2.CreateGameRequest(
        config=game_pb2.GameConfig(
            width=20,
            height=20,
            max_players=2,
            fog_of_war=True,
            turn_time_ms=5000
        )
    )
    response = stub.CreateGame(request)
    print(f"Created game: {response.game_id}")
    return response.game_id

def join_game(stub, game_id, player_name):
    """Join an existing game."""
    request = game_pb2.JoinGameRequest(
        game_id=game_id,
        player_name=player_name
    )
    response = stub.JoinGame(request)
    print(f"Joined as player {response.player_id} with token {response.player_token}")
    return response.player_id, response.player_token, response.initial_state

def main():
    # Connect to the gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = game_pb2_grpc.GameServiceStub(channel)
    
    try:
        # Create a new game
        game_id = create_game(stub)
        
        # Join as player 1
        player_id, token, state = join_game(stub, game_id, "Player1")
        print(f"Initial state: Turn {state.turn}, Board {state.board.width}x{state.board.height}")
        
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()}: {e.details()}")
    finally:
        channel.close()

if __name__ == "__main__":
    main()
EOF

chmod +x python/examples/simple_client.py

echo -e "${GREEN}Python proto generation complete!${NC}"
echo ""
echo "Generated files in:"
echo "  - python/generals_pb/common/v1/"
echo "  - python/generals_pb/game/v1/"
echo "  - python/generals_pb/experience/v1/"
echo ""
echo "To use the Python client:"
echo "  1. cd python"
echo "  2. pip install -r requirements.txt"
echo "  3. python examples/simple_client.py"