#!/usr/bin/env python3
"""
Simple test to verify experience collection is working end-to-end.

This test:
1. Creates a game
2. Joins as two players
3. Plays a few turns with random actions
4. Verifies experiences are being collected
"""

import time
import grpc

from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2
from generals_agent.experience_consumer import ExperienceConsumer


def test_experience_collection(server_address="localhost:50051"):
    """Test that experience collection works end-to-end."""
    print(f"Testing experience collection on {server_address}\n")
    
    # Connect to game server
    channel = grpc.insecure_channel(server_address)
    game_stub = game_pb2_grpc.GameServiceStub(channel)
    
    # Create experience consumer
    exp_consumer = ExperienceConsumer(server_address)
    
    try:
        # 1. Create a new game
        print("1. Creating game...")
        create_req = game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(
                width=10,
                height=10,
                max_players=2,
                fog_of_war=True,
                turn_time_ms=0,  # No turn timeout for testing
                collect_experiences=True  # Enable experience collection
            )
        )
        create_resp = game_stub.CreateGame(create_req)
        game_id = create_resp.game_id
        print(f"   Created game: {game_id}")
        
        # 2. Join as two players
        print("\n2. Joining players...")
        players = []
        for i in range(2):
            join_req = game_pb2.JoinGameRequest(
                game_id=game_id,
                player_name=f"TestPlayer{i+1}"
            )
            join_resp = game_stub.JoinGame(join_req)
            players.append({
                'id': join_resp.player_id,
                'token': join_resp.player_token,
                'name': f"TestPlayer{i+1}"
            })
            print(f"   Player {i+1} joined with ID {join_resp.player_id}")
        
        # 3. Game starts automatically when players join
        print("\n3. Game started automatically after players joined")
        
        # Get initial experience count
        initial_stats = exp_consumer.get_experience_stats([game_id])
        initial_count = initial_stats.total_experiences
        print(f"\n4. Initial experience count: {initial_count}")
        
        # First, advance turn 0 to trigger initial production
        # In turn 0, generals start with 1 army and need production to get to 2
        print("\n   Advancing turn 0 to trigger initial production...")
        
        # Submit no-op actions for both players to advance turn
        for player in players:
            # For turn 0, submit a "no-op" action since players can't move with only 1 army
            action = game_pb2.Action(
                type=common_pb2.ACTION_TYPE_UNSPECIFIED,
                turn_number=0,
                half=False
            )
            
            action_req = game_pb2.SubmitActionRequest(
                game_id=game_id,
                player_id=player['id'],
                player_token=player['token'],
                action=action
            )
            try:
                game_stub.SubmitAction(action_req)
            except grpc.RpcError as e:
                print(f"     Warning: Turn 0 action for {player['name']} failed: {e.details()}")
        
        # Small delay to ensure turn processing completes
        time.sleep(0.5)
        
        # Get updated state to see army counts
        state_req = game_pb2.GetGameStateRequest(
            game_id=game_id,
            player_id=players[0]['id'],
            player_token=players[0]['token']
        )
        state_resp = game_stub.GetGameState(state_req)
        print(f"\n   Current turn: {state_resp.state.turn}")
        print(f"   Game phase: {state_resp.state.current_phase}")
        
        # 5. Play some turns
        print("\n5. Playing 10 turns...")
        for turn in range(10):
            print(f"\n   Turn {turn + 1}:")
            
            # Collect actions for all players first
            player_actions = []
            
            # Each player finds a move
            for player in players:
                # Get game state
                state_req = game_pb2.GetGameStateRequest(
                    game_id=game_id,
                    player_id=player['id'],
                    player_token=player['token']
                )
                state_resp = game_stub.GetGameState(state_req)
                
                # Debug: Print player's tiles
                owned_tiles = []
                
                # Find a valid move (just move from first owned tile)
                action = None
                if state_resp.state.players:
                    for y in range(state_resp.state.board.height):
                        for x in range(state_resp.state.board.width):
                            idx = y * state_resp.state.board.width + x
                            tile = state_resp.state.board.tiles[idx]
                            if tile.owner_id == player['id']:
                                owned_tiles.append((x, y, tile.army_count))
                            if tile.owner_id == player['id'] and tile.army_count > 1:
                                # Try to move in any direction
                                directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
                                for dx, dy in directions:
                                    new_x, new_y = x + dx, y + dy
                                    if 0 <= new_x < state_resp.state.board.width and 0 <= new_y < state_resp.state.board.height:
                                        action = game_pb2.Action(
                                            type=common_pb2.ACTION_TYPE_MOVE,
                                            turn_number=state_resp.state.turn,
                                            half=False
                                        )
                                        getattr(action, 'from').CopyFrom(common_pb2.Coordinate(x=x, y=y))
                                        action.to.CopyFrom(common_pb2.Coordinate(x=new_x, y=new_y))
                                        break
                            if action:
                                break
                        if action:
                            break
                
                print(f"     Player {player['name']} owns {len(owned_tiles)} tiles: {owned_tiles[:5]}...")
                
                # If no valid move, create a no-op action
                if not action:
                    action = game_pb2.Action(
                        type=common_pb2.ACTION_TYPE_UNSPECIFIED,
                        turn_number=state_resp.state.turn,
                        half=False
                    )
                    print(f"     Player {player['name']} has no valid moves, submitting no-op")
                else:
                    print(f"     Player {player['name']} will move from ({getattr(action, 'from').x}, {getattr(action, 'from').y}) to ({action.to.x}, {action.to.y})")
                
                player_actions.append((player, action))
            
            # Now submit all actions to trigger turn processing
            for player, action in player_actions:
                action_req = game_pb2.SubmitActionRequest(
                    game_id=game_id,
                    player_id=player['id'],
                    player_token=player['token'],
                    action=action
                )
                try:
                    action_resp = game_stub.SubmitAction(action_req)
                except grpc.RpcError as e:
                    print(f"     Player {player['name']} action submission failed: {e.details()}")
            
            # Small delay to ensure processing
            time.sleep(0.5)
        
        # 5. Check if experiences were collected
        print("\n6. Checking collected experiences...")
        time.sleep(1)  # Give time for experiences to be processed
        
        final_stats = exp_consumer.get_experience_stats([game_id])
        final_count = final_stats.total_experiences
        experiences_collected = final_count - initial_count
        
        print(f"   Experiences collected: {experiences_collected}")
        print(f"   Total experiences now: {final_count}")
        
        if experiences_collected > 0:
            print("\n✅ SUCCESS: Experience collection is working!")
            
            # Try to stream some experiences
            print("\n7. Streaming collected experiences...")
            exp_count = 0
            for batch in exp_consumer.stream_experiences(
                game_ids=[game_id],
                batch_size=5,
                follow=False
            ):
                for exp in batch:
                    exp_count += 1
                    print(f"   Experience {exp_count}: "
                          f"Player {exp.metadata['player_id']}, "
                          f"Turn {exp.metadata['turn']}, "
                          f"Reward {exp.reward:.3f}")
                    
                    if exp_count >= 5:  # Just show first 5
                        break
                if exp_count >= 5:
                    break
            
            return True
        else:
            print("\n❌ FAILURE: No experiences were collected!")
            return False
            
    except grpc.RpcError as e:
        print(f"\n❌ gRPC Error: {e.code()}: {e.details()}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        channel.close()
        exp_consumer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test experience collection")
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server address"
    )
    
    args = parser.parse_args()
    
    success = test_experience_collection(args.server)
    sys.exit(0 if success else 1)