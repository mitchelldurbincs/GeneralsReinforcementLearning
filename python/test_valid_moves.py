#!/usr/bin/env python3
"""
Test script that generates valid moves by parsing the game board.
"""

import sys
import os
import grpc
import numpy as np

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
        if tile.owner_id >= 0 and tile.army_count > 0:  # Player owns this tile and has armies
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


def test_with_valid_moves():
    """Test experience streaming with valid moves."""
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
    
    # Process 10 turns with valid moves
    for turn in range(10):
        print(f"\nTurn {turn + 1}:")
        
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
            
            # Debug: print what tiles each player owns
            if turn == 0:
                print(f"  Player {player_id} owns {len(player_tiles.get(player_id, []))} tiles:")
                for pid, tiles in player_tiles.items():
                    if pid == player_id:
                        for tile in tiles[:3]:  # Show first 3 tiles
                            print(f"    Tile at ({tile['x']},{tile['y']}): army={tile['army']}, type={tile['type']}")
            
            # Get valid moves
            valid_moves = get_valid_moves(board, player_id, player_tiles)
            
            if valid_moves:
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
                        print(f"  Player {player_id}: moved from ({move['from_x']},{move['from_y']}) to ({move['to_x']},{move['to_y']})")
                    else:
                        print(f"  Player {player_id}: move failed - {submit_resp.error_message}")
                except grpc.RpcError as e:
                    print(f"  Player {player_id}: error - {e.code()}")
            else:
                print(f"  Player {player_id}: no valid moves available")
                
                # Submit a pass action (no move)
                # For now, just skip
    
    # Check if experiences were collected
    stats_req = experience_pb2.GetExperienceStatsRequest(game_ids=[game_id])
    stats_resp = exp_stub.GetExperienceStats(stats_req)
    
    print(f"\n{'='*50}")
    print(f"Experiences collected: {stats_resp.total_experiences}")
    if stats_resp.total_experiences > 0:
        print(f"Average reward: {stats_resp.average_reward:.4f}")
        print("✓ Experience collection is working!")
        return True
    else:
        print("✗ No experiences collected")
        return False


if __name__ == "__main__":
    success = test_with_valid_moves()
    sys.exit(0 if success else 1)