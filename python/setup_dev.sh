#!/bin/bash

# Development setup script for Generals RL Python package
# This script sets up the development environment and installs the package in editable mode

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Generals RL Python development environment...${NC}"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/generalsrl" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv "$PROJECT_ROOT/generalsrl"
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$PROJECT_ROOT/generalsrl/bin/activate"

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install the package in editable mode
echo -e "${GREEN}Installing generals-rl package in editable mode...${NC}"
cd "$SCRIPT_DIR"
pip install -e .

# Install development dependencies
echo -e "${GREEN}Installing development dependencies...${NC}"
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
fi

# Export PYTHONPATH for the current session
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${YELLOW}Note: The package has been installed in editable mode.${NC}"
echo -e "${YELLOW}To use this environment in a new terminal, run:${NC}"
echo -e "  source $PROJECT_ROOT/generalsrl/bin/activate"
echo -e "  export PYTHONPATH=$SCRIPT_DIR:\$PYTHONPATH"
echo ""
echo -e "${GREEN}You can now import the package with:${NC}"
echo -e "  from generals_agent import BaseAgent"
echo -e "  from generals_pb.game.v1 import game_pb2"