#!/usr/bin/env python3
"""
Test script to debug board state directly.
"""

import sys
import os
import grpc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2


def test_board_debug():
    """Debug board state."""
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
    
    # Join as player 0
    join_req = game_pb2.JoinGameRequest(
        game_id=game_id,
        player_name="TestPlayer0"
    )
    join_resp = game_stub.JoinGame(join_req)
    player0_id = join_resp.player_id
    player0_token = join_resp.player_token
    print(f"Joined as player {player0_id}, status: {join_resp.initial_state.status}")
    
    # Join as player 1 - this starts the game
    join_req = game_pb2.JoinGameRequest(
        game_id=game_id,
        player_name="TestPlayer1"
    )
    join_resp = game_stub.JoinGame(join_req)
    player1_id = join_resp.player_id
    player1_token = join_resp.player_token
    print(f"Joined as player {player1_id}, status: {join_resp.initial_state.status}")
    
    # Now check the board state - should have 2 armies at generals
    board = join_resp.initial_state.board
    width = board.width
    
    generals_found = []
    for idx, tile in enumerate(board.tiles):
        if tile.type == common_pb2.TileType.TILE_TYPE_GENERAL:
            y = idx // width
            x = idx % width
            generals_found.append({
                'x': x,
                'y': y,
                'owner': tile.owner_id,
                'army': tile.army_count
            })
    
    print(f"\nGenerals found: {len(generals_found)}")
    for g in generals_found:
        print(f"  Position ({g['x']},{g['y']}): owner={g['owner']}, army={g['army']}")
    
    # Check if armies are correct
    expected_armies = 2  # We set this in generator.go
    all_correct = all(g['army'] == expected_armies for g in generals_found if g['owner'] >= 0)
    
    if all_correct:
        print(f"\n✓ All generals have {expected_armies} armies as expected!")
        return True
    else:
        print(f"\n✗ Generals should have {expected_armies} armies but they don't")
        print("\nThis indicates the change in generator.go may not have taken effect")
        return False


if __name__ == "__main__":
    success = test_board_debug()
    sys.exit(0 if success else 1)