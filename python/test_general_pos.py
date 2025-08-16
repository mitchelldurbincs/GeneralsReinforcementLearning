#!/usr/bin/env python3
"""Test general position fix"""

from generals_agent import RandomAgent, GameConfig, GameConnection, GameClient
from generals_pb.game.v1 import game_pb2
from generals_pb.common.v1 import common_pb2

# Create a test game state
def create_test_state():
    state = game_pb2.GameState()
    state.turn = 0
    state.status = common_pb2.GAME_STATUS_IN_PROGRESS
    state.board.width = 10
    state.board.height = 10
    
    # Create tiles
    for i in range(100):
        tile = state.board.tiles.add()
        tile.type = common_pb2.TILE_TYPE_NORMAL
        tile.owner_id = -1
        tile.army_count = 0
        tile.visible = True
        tile.fog_of_war = False
    
    # Set a general for player 0 at position (5, 5)
    idx = 5 * 10 + 5
    state.board.tiles[idx].type = common_pb2.TILE_TYPE_GENERAL
    state.board.tiles[idx].owner_id = 0
    state.board.tiles[idx].army_count = 1
    
    # Add player
    player = state.players.add()
    player.id = 0
    player.name = "TestPlayer"
    player.status = common_pb2.PLAYER_STATUS_ACTIVE
    player.army_count = 1
    player.tile_count = 1
    
    return state

# Test the agent
agent = RandomAgent(name="TestAgent")
agent.player_id = 0

state = create_test_state()
print("Testing on_game_start...")
try:
    agent.on_game_start(state)
    print(f"Success! General position: {agent.general_position}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()