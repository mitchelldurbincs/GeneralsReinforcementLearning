#!/usr/bin/env python3
"""Simple test to check if StreamGame works"""

import grpc
import logging
import time
from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = game_pb2_grpc.GameServiceStub(channel)
    
    # Create game
    create_resp = stub.CreateGame(game_pb2.CreateGameRequest(
        config=game_pb2.GameConfig(width=10, height=10, max_players=2)
    ))
    game_id = create_resp.game_id
    logger.info(f"Created game: {game_id}")
    
    # Join as player 1
    join1 = stub.JoinGame(game_pb2.JoinGameRequest(
        game_id=game_id, player_name="Player1"
    ))
    logger.info(f"Player 1 joined: ID={join1.player_id}")
    
    # Join as player 2
    join2 = stub.JoinGame(game_pb2.JoinGameRequest(
        game_id=game_id, player_name="Player2"
    ))
    logger.info(f"Player 2 joined: ID={join2.player_id}")
    
    # Start streaming for player 1
    stream_req = game_pb2.StreamGameRequest(
        game_id=game_id,
        player_id=join1.player_id,
        player_token=join1.player_token
    )
    
    logger.info("Starting stream...")
    stream = stub.StreamGame(stream_req)
    
    # Get first update
    try:
        update = next(stream)
        logger.info("Got first update!")
        
        if update.HasField('full_state'):
            state = update.full_state
            logger.info(f"State: Turn {state.turn}, Status: {state.status}, Phase: {state.current_phase}")
            
            # Submit an action to trigger turn processing
            logger.info("Submitting action to trigger turn...")
            stub.SubmitAction(game_pb2.SubmitActionRequest(
                game_id=game_id,
                player_id=join1.player_id,
                player_token=join1.player_token,
                action=game_pb2.Action(
                    type=common_pb2.ACTION_TYPE_UNSPECIFIED,
                    turn_number=state.turn
                )
            ))
            
            # Also submit for player 2
            stub.SubmitAction(game_pb2.SubmitActionRequest(
                game_id=game_id,
                player_id=join2.player_id,
                player_token=join2.player_token,
                action=game_pb2.Action(
                    type=common_pb2.ACTION_TYPE_UNSPECIFIED,
                    turn_number=state.turn
                )
            ))
            
            # Get next update
            logger.info("Waiting for next update after actions...")
            update2 = next(stream)
            logger.info(f"Got second update! Type: {update2.WhichOneof('update')}")
            
    except StopIteration:
        logger.error("Stream ended unexpectedly")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()