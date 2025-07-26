#!/usr/bin/env python3
"""
Test script to explore gRPC functionality with the Generals game server.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import grpc
import time
import random
from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2

def test_server_connection(stub):
    """Test basic server connectivity."""
    print("Testing server connection...")
    try:
        # Try to create a game to test connectivity
        request = game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(
                width=10,
                height=10,
                max_players=2,
                fog_of_war=True,
                turn_time_ms=1000
            )
        )
        response = stub.CreateGame(request)
        print(f"✓ Server is responsive. Created test game: {response.game_id}")
        return response.game_id
    except grpc.RpcError as e:
        print(f"✗ Server connection failed: {e.code()}: {e.details()}")
        return None

def test_game_lifecycle(stub, game_id):
    """Test the full game lifecycle."""
    print("\nTesting game lifecycle...")
    
    # Join as player 1
    join_request = game_pb2.JoinGameRequest(
        game_id=game_id,
        player_name="TestPlayer1"
    )
    join_response = stub.JoinGame(join_request)
    player1_id = join_response.player_id
    player1_token = join_response.player_token
    
    print(f"✓ Player 1 joined: ID={player1_id}")
    print(f"  Initial state: Turn {join_response.initial_state.turn}")
    print(f"  Board size: {join_response.initial_state.board.width}x{join_response.initial_state.board.height}")
    
    # Join as player 2
    join_request2 = game_pb2.JoinGameRequest(
        game_id=game_id,
        player_name="TestPlayer2"
    )
    join_response2 = stub.JoinGame(join_request2)
    player2_id = join_response2.player_id
    player2_token = join_response2.player_token
    
    print(f"✓ Player 2 joined: ID={player2_id}")
    
    # Get game state
    state_request = game_pb2.GetGameStateRequest(
        game_id=game_id,
        player_token=player1_token
    )
    state_response = stub.GetGameState(state_request)
    
    print(f"\n✓ Got game state:")
    print(f"  Status: {common_pb2.GameStatus.Name(state_response.state.status)}")
    print(f"  Turn: {state_response.state.turn}")
    print(f"  Number of tiles: {len(state_response.state.board.tiles)}")
    
    # Find player's general position
    general_pos = None
    board_width = state_response.state.board.width
    board_height = state_response.state.board.height
    
    for idx, tile in enumerate(state_response.state.board.tiles):
        if tile.type == common_pb2.TILE_TYPE_GENERAL and tile.owner_id == player1_id:
            # Calculate x,y from flattened index (row-major order)
            y = idx // board_width
            x = idx % board_width
            general_pos = (x, y)
            print(f"  Player 1 general at: ({x}, {y})")
            break
    
    # Submit a test move
    if general_pos and state_response.state.status == common_pb2.GAME_STATUS_IN_PROGRESS:
        print("\nTesting move submission...")
        
        # Try to move from general to an adjacent tile
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = random.choice(directions)
        target_x = general_pos[0] + dx
        target_y = general_pos[1] + dy
        
        move_request = game_pb2.SubmitActionRequest(
            game_id=game_id,
            player_token=player1_token,
            action=game_pb2.Action(
                type=common_pb2.ACTION_TYPE_MOVE,
                **{'from': common_pb2.Coordinate(x=general_pos[0], y=general_pos[1])},
                to=common_pb2.Coordinate(x=target_x, y=target_y)
            )
        )
        
        try:
            move_response = stub.SubmitAction(move_request)
            if move_response.success:
                print(f"✓ Move submitted successfully")
            else:
                print(f"✗ Move failed: {move_response.error_message}")
        except grpc.RpcError as e:
            print(f"✗ Move submission error: {e.code()}: {e.details()}")
        except Exception as e:
            print(f"✗ Move submission error: {type(e).__name__}: {e}")
    
    return player1_token, player2_token

def test_streaming(stub, game_id, player_token):
    """Test game update streaming."""
    print("\nTesting game update streaming...")
    
    stream_request = game_pb2.StreamGameUpdatesRequest(
        game_id=game_id,
        player_token=player_token
    )
    
    try:
        # Stream updates for a few seconds
        print("Listening for game updates (5 seconds)...")
        stream = stub.StreamGameUpdates(stream_request)
        
        start_time = time.time()
        update_count = 0
        
        for update in stream:
            update_count += 1
            print(f"  Update {update_count}: Turn {update.state.turn}, Status: {common_pb2.GameStatus.Name(update.state.status)}")
            
            if time.time() - start_time > 5:
                print("✓ Streaming test complete")
                break
                
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.CANCELLED:
            print("✓ Stream cancelled")
        else:
            print(f"✗ Streaming error: {e.code()}: {e.details()}")

def main():
    # Connect to the gRPC server
    print("Connecting to Generals game server at localhost:50051...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = game_pb2_grpc.GameServiceStub(channel)
    
    try:
        # Test 1: Server connection
        game_id = test_server_connection(stub)
        if not game_id:
            print("\nCannot proceed without server connection.")
            print("Make sure the game server is running: go run cmd/game_server/main.go")
            return
        
        # Test 2: Game lifecycle
        player1_token, player2_token = test_game_lifecycle(stub, game_id)
        
        # Test 3: Streaming (commented out for now as it may block)
        # test_streaming(stub, game_id, player1_token)
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
    finally:
        channel.close()

if __name__ == "__main__":
    main()