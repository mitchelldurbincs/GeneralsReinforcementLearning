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
