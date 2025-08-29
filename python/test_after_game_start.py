#!/usr/bin/env python3
"""
Test to check board state after game starts.
"""

import sys
import os
import grpc
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2


def test_after_start():
    """Check board state after game starts."""
    channel = grpc.insecure_channel("localhost:50051")
    game_stub = game_pb2_grpc.GameServiceStub(channel)
    
    # Create a game
    create_req = game_pb2.CreateGameRequest(
        config=game_pb2.GameConfig(
            width=10,
            height=10,
            max_players=2,
            fog_of_war=False,
            collect_experiences=True
        )
    )
    
    create_resp = game_stub.CreateGame(create_req)
    game_id = create_resp.game_id
    print(f"Created game: {game_id}")
    
    # Join as both players
    players = {}
    for i in range(2):
        join_req = game_pb2.JoinGameRequest(
            game_id=game_id,
            player_name=f"TestPlayer{i}"
        )
        join_resp = game_stub.JoinGame(join_req)
        players[join_resp.player_id] = {
            'id': join_resp.player_id,
            'token': join_resp.player_token,
        }
        print(f"Joined as player {join_resp.player_id}")
        print(f"  Initial state status: {join_resp.initial_state.status}")
        
        # Check board state from join response
        board = join_resp.initial_state.board
        if board and board.tiles:
            # Find generals
            width = board.width
            for idx, tile in enumerate(board.tiles):
                if tile.type == common_pb2.TileType.TILE_TYPE_GENERAL:
                    y = idx // width
                    x = idx % width
                    print(f"  General at ({x},{y}): owner={tile.owner_id}, army={tile.army_count}")
    
    print("\n--- After game starts, checking actual state ---")
    
    # Now get the actual game state after it's started
    for player_id, player_info in players.items():
        state_req = game_pb2.GetGameStateRequest(
            game_id=game_id,
            player_id=player_id,
            player_token=player_info['token']
        )
        
        state_resp = game_stub.GetGameState(state_req)
        print(f"\nPlayer {player_id} view:")
        print(f"  Game status: {state_resp.state.status}")
        print(f"  Turn: {state_resp.state.turn}")
        
        board = state_resp.state.board
        width = board.width
        
        # Find and show generals
        for idx, tile in enumerate(board.tiles):
            if tile.type == common_pb2.TileType.TILE_TYPE_GENERAL:
                y = idx // width
                x = idx % width
                print(f"  General at ({x},{y}): owner={tile.owner_id}, army={tile.army_count}")
        
        # Break after first player to avoid redundancy
        break
    
    return True


if __name__ == "__main__":
    success = test_after_start()
    sys.exit(0 if success else 1)