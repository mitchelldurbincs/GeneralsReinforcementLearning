#!/usr/bin/env python3
"""
Test script that waits for production before generating moves.
Players start with 1 army at their general, but need 2 to move.
Production happens every turn for generals/cities.
"""

import sys
import os
import grpc
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.experience.v1 import experience_pb2, experience_pb2_grpc
from generals_pb.common.v1 import common_pb2


def parse_board(board_state):
    """Parse the board to find owned tiles for each player."""
    width = board_state.width
    height = board_state.height
    tiles = board_state.tiles
    
    player_tiles = {}
    
    for idx, tile in enumerate(tiles):
        if tile.owner_id >= 0:  # Player owns this tile
            y = idx // width
            x = idx % width
            
            if tile.owner_id not in player_tiles:
                player_tiles[tile.owner_id] = []
            player_tiles[tile.owner_id].append({
                'x': x,
                'y': y,
                'army': tile.army_count,
                'type': tile.type,
                'idx': idx
            })
    
    return player_tiles


def get_valid_moves(board_state, player_id, player_tiles):
    """Get valid moves for a player based on their owned tiles."""
    width = board_state.width
    height = board_state.height
    valid_moves = []
    
    if player_id not in player_tiles:
        return valid_moves
    
    for tile in player_tiles[player_id]:
        if tile['army'] <= 1:  # Need at least 2 armies to move (1 stays behind)
            continue
            
        x, y = tile['x'], tile['y']
        
        # Check all 4 adjacent tiles
        adjacent = [
            (x, y-1),  # up
            (x, y+1),  # down  
            (x-1, y),  # left
            (x+1, y),  # right
        ]
        
        for nx, ny in adjacent:
            # Check bounds
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
                
            # This is a valid move
            valid_moves.append({
                'from_x': x,
                'from_y': y,
                'to_x': nx,
                'to_y': ny,
                'army': tile['army']
            })
    
    return valid_moves


def test_with_production():
    """Test experience streaming by waiting for production."""
    channel = grpc.insecure_channel("localhost:50051")
    game_stub = game_pb2_grpc.GameServiceStub(channel)
    exp_stub = experience_pb2_grpc.ExperienceServiceStub(channel)
    
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
            'initial_state': join_resp.initial_state
        }
        print(f"Joined as player {join_resp.player_id}")
    
    # Track when we first get valid moves
    first_valid_turn = -1
    
    # Process turns until we have enough armies to move
    for turn in range(50):  # Max 50 turns
        print(f"\nTurn {turn + 1}:")
        
        actions_submitted = 0
        
        # Get current state and find valid moves for each player
        for player_id, player_info in players.items():
            state_req = game_pb2.GetGameStateRequest(
                game_id=game_id,
                player_id=player_id,
                player_token=player_info['token']
            )
            
            state_resp = game_stub.GetGameState(state_req)
            board = state_resp.state.board
            
            # Parse board to find owned tiles
            player_tiles = parse_board(board)
            
            # Debug: Show army counts for first few turns
            if turn < 3 and player_id in player_tiles:
                print(f"  Player {player_id} tiles:")
                for tile in player_tiles[player_id][:3]:  # Show first 3 tiles
                    print(f"    ({tile['x']},{tile['y']}): army={tile['army']}, type={tile['type']}")
            
            # Get valid moves
            valid_moves = get_valid_moves(board, player_id, player_tiles)
            
            if valid_moves:
                if first_valid_turn == -1:
                    first_valid_turn = turn + 1
                    print(f"  >>> First valid moves available on turn {first_valid_turn}!")
                
                # Pick a random valid move
                move = np.random.choice(valid_moves)
                
                # Create and submit action
                action = game_pb2.Action()
                action.type = common_pb2.ActionType.ACTION_TYPE_MOVE
                action.to.x = move['to_x']
                action.to.y = move['to_y']
                action.turn_number = state_resp.state.turn
                action.half = False
                
                from_coord = getattr(action, 'from')
                from_coord.x = move['from_x']
                from_coord.y = move['from_y']
                
                submit_req = game_pb2.SubmitActionRequest(
                    game_id=game_id,
                    player_id=player_id,
                    player_token=player_info['token'],
                    action=action
                )
                
                try:
                    submit_resp = game_stub.SubmitAction(submit_req)
                    if submit_resp.success:
                        actions_submitted += 1
                        print(f"  Player {player_id}: moved from ({move['from_x']},{move['from_y']}) to ({move['to_x']},{move['to_y']})")
                    else:
                        print(f"  Player {player_id}: move failed - {submit_resp.error_message}")
                except grpc.RpcError as e:
                    print(f"  Player {player_id}: error - {e.code()}")
            else:
                # No valid moves yet - submit a null move (from and to are same position)
                # Find our general position
                if player_id in player_tiles and player_tiles[player_id]:
                    # Use first owned tile (should be general)
                    tile = player_tiles[player_id][0]
                    
                    action = game_pb2.Action()
                    action.type = common_pb2.ActionType.ACTION_TYPE_MOVE
                    action.to.x = tile['x']
                    action.to.y = tile['y']
                    action.turn_number = state_resp.state.turn
                    action.half = False
                    
                    from_coord = getattr(action, 'from')
                    from_coord.x = tile['x']
                    from_coord.y = tile['y']
                    
                    submit_req = game_pb2.SubmitActionRequest(
                        game_id=game_id,
                        player_id=player_id,
                        player_token=player_info['token'],
                        action=action
                    )
                    
                    try:
                        submit_resp = game_stub.SubmitAction(submit_req)
                        if submit_resp.success:
                            print(f"  Player {player_id}: null move at ({tile['x']},{tile['y']}) (waiting for production)")
                        else:
                            print(f"  Player {player_id}: null move failed - {submit_resp.error_message}")
                    except grpc.RpcError as e:
                        print(f"  Player {player_id}: null move error - {e.code()}")
                else:
                    print(f"  Player {player_id}: no tiles owned?")
        
        # If we've had valid moves for a few turns, we can check experiences
        if first_valid_turn > 0 and turn >= first_valid_turn + 5:
            break
    
    # Check if experiences were collected
    print(f"\n{'='*50}")
    print(f"Game completed after {turn + 1} turns")
    print(f"First valid moves available on turn {first_valid_turn}")
    
    # Give server a moment to process
    time.sleep(0.5)
    
    stats_req = experience_pb2.GetExperienceStatsRequest(game_ids=[game_id])
    stats_resp = exp_stub.GetExperienceStats(stats_req)
    
    print(f"\nExperiences collected: {stats_resp.total_experiences}")
    if stats_resp.total_experiences > 0:
        print(f"Average reward: {stats_resp.average_reward:.4f}")
        print("✓ Experience collection is working!")
        
        # Try to stream some experiences
        stream_req = experience_pb2.StreamExperienceBatchesRequest(
            game_ids=[game_id],
            batch_size=10,
            follow=False
        )
        
        print("\nStreaming experience batches:")
        batch_count = 0
        exp_count = 0
        
        for batch in exp_stub.StreamExperienceBatches(stream_req):
            batch_count += 1
            exp_count += len(batch.experiences)
            if batch_count <= 2:  # Show first 2 batches
                print(f"  Batch {batch_count}: {len(batch.experiences)} experiences")
                if batch.experiences:
                    exp = batch.experiences[0]
                    print(f"    Sample: Turn {exp.turn}, Player {exp.player_id}, Reward {exp.reward:.2f}")
        
        print(f"Total: {batch_count} batches, {exp_count} experiences")
        return True
    else:
        print("✗ No experiences collected")
        print("\nPossible reasons:")
        print("- Game may not have progressed enough")
        print("- Check server logs for errors")
        return False


if __name__ == "__main__":
    success = test_with_production()
    sys.exit(0 if success else 1)