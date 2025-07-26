#!/usr/bin/env python3
"""Test a single random agent with verbose logging"""

import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generals_agent.random_agent import RandomAgent

# Set up very verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create and run a single agent
agent = RandomAgent(agent_name="TestAgent")
agent.connect()

# Create a new game
game_id = agent.create_game(width=10, height=10)
print(f"Created game: {game_id}")

# Join and play
try:
    agent.join_game(game_id)
    print(f"Joined as player {agent.player_id}")
    
    # Get initial state
    state = agent.get_game_state()
    print(f"Initial state - Turn: {state.turn}, Board size: {state.board.width}x{state.board.height}")
    print(f"Number of tiles: {len(state.board.tiles)}")
    
    # Try streaming
    agent.on_game_start()
    agent.stream_game_updates()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    agent.disconnect()