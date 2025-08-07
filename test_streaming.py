#!/usr/bin/env python3
"""Simple test to verify StreamGame functionality"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import grpc
import logging
import time
from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_streaming():
    """Test that StreamGame endpoint works"""
    
    # Connect to server
    channel = grpc.insecure_channel('localhost:50051')
    stub = game_pb2_grpc.GameServiceStub(channel)
    
    try:
        # Create a game
        create_req = game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(
                width=10,
                height=10,
                max_players=2,
                fog_of_war=False,
                turn_time_ms=500
            )
        )
        create_resp = stub.CreateGame(create_req)
        game_id = create_resp.game_id
        logger.info(f"Created game: {game_id}")
        
        # Join the game as player 1
        join_req1 = game_pb2.JoinGameRequest(
            game_id=game_id,
            player_name="TestPlayer1"
        )
        join_resp1 = stub.JoinGame(join_req1)
        player1_id = join_resp1.player_id
        player1_token = join_resp1.player_token
        logger.info(f"Player 1 joined: ID={player1_id}")
        
        # Join as player 2
        join_req2 = game_pb2.JoinGameRequest(
            game_id=game_id,
            player_name="TestPlayer2"
        )
        join_resp2 = stub.JoinGame(join_req2)
        player2_id = join_resp2.player_id
        player2_token = join_resp2.player_token
        logger.info(f"Player 2 joined: ID={player2_id}")
        
        # Test streaming for player 1
        stream_req = game_pb2.StreamGameRequest(
            game_id=game_id,
            player_id=player1_id,
            player_token=player1_token
        )
        
        logger.info("Connecting to stream...")
        stream = stub.StreamGame(stream_req)
        
        # Read a few updates
        update_count = 0
        for update in stream:
            update_count += 1
            logger.info(f"Received update #{update_count}")
            
            if update.HasField('full_state'):
                state = update.full_state
                logger.info(f"  Full state: Turn {state.turn}, Status: {common_pb2.GameStatus.Name(state.status)}")
                logger.info(f"  Board size: {state.board.width}x{state.board.height}")
                logger.info(f"  Players: {len(state.players)}")
            elif update.HasField('delta'):
                delta = update.delta
                logger.info(f"  Delta update: Turn {delta.turn}, {len(delta.tile_updates)} tile changes")
            elif update.HasField('event'):
                event = update.event
                if event.HasField('game_started'):
                    logger.info("  Event: Game started!")
                elif event.HasField('phase_changed'):
                    phase = event.phase_changed
                    logger.info(f"  Event: Phase changed from {phase.previous_phase} to {phase.new_phase}")
            
            # Stop after a few updates to avoid infinite loop
            if update_count >= 3:
                logger.info("Test successful! Streaming works.")
                break
                
        return True
        
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        channel.close()

if __name__ == "__main__":
    # First check if server is running
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        # Try to call a simple method
        stub.CreateGame(game_pb2.CreateGameRequest(), timeout=1)
    except:
        logger.error("Server not running! Start the server with: go run cmd/game_server/main.go")
        sys.exit(1)
    
    success = test_streaming()
    sys.exit(0 if success else 1)