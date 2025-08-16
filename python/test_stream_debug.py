#!/usr/bin/env python3
"""Debug streaming issue"""

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
    logger.info("Creating game...")
    create_resp = stub.CreateGame(game_pb2.CreateGameRequest(
        config=game_pb2.GameConfig(width=10, height=10, max_players=2)
    ))
    game_id = create_resp.game_id
    logger.info(f"Created game: {game_id}")
    
    # Join as player 1
    logger.info("Joining as player 1...")
    join1 = stub.JoinGame(game_pb2.JoinGameRequest(
        game_id=game_id, player_name="Player1"
    ))
    logger.info(f"Player 1 joined: ID={join1.player_id}")
    
    # Join as player 2 with timeout
    logger.info("Joining as player 2...")
    try:
        join2 = stub.JoinGame(game_pb2.JoinGameRequest(
            game_id=game_id, player_name="Player2"
        ), timeout=5)
        logger.info(f"Player 2 joined: ID={join2.player_id}")
    except grpc.RpcError as e:
        logger.error(f"Failed to join as player 2: {e.code()} - {e.details()}")
        return
    
    # Now test streaming
    logger.info("Starting stream for player 1...")
    stream_req = game_pb2.StreamGameRequest(
        game_id=game_id,
        player_id=join1.player_id,
        player_token=join1.player_token
    )
    
    stream = stub.StreamGame(stream_req)
    
    # Get first update with timeout
    logger.info("Waiting for first update...")
    try:
        import itertools
        for i, update in enumerate(itertools.islice(stream, 3)):
            logger.info(f"Got update {i+1}")
            if update.HasField('full_state'):
                state = update.full_state
                logger.info(f"  Full state: Turn {state.turn}, Phase: {state.current_phase}")
            elif update.HasField('event'):
                logger.info(f"  Event received")
            elif update.HasField('delta'):
                logger.info(f"  Delta update")
    except Exception as e:
        logger.error(f"Stream error: {e}")

if __name__ == "__main__":
    main()