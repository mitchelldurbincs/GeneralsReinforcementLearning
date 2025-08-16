#!/usr/bin/env python3
"""
Simple test to verify experience collection is working.
"""

import sys
import os
import time
import grpc

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.experience.v1 import experience_pb2, experience_pb2_grpc

def main():
    print("=== Simple Experience Collection Test ===")
    
    # Connect to server
    channel = grpc.insecure_channel('localhost:50051')
    game_stub = game_pb2_grpc.GameServiceStub(channel)
    exp_stub = experience_pb2_grpc.ExperienceServiceStub(channel)
    
    # Create a game with experience collection enabled
    create_req = game_pb2.CreateGameRequest(
        config=game_pb2.GameConfig(
            width=10,
            height=10,
            max_players=2,
            collect_experiences=True
        )
    )
    
    try:
        create_resp = game_stub.CreateGame(create_req)
        game_id = create_resp.game_id
        print(f"✓ Created game: {game_id}")
        print(f"  Experience collection: ENABLED")
    except grpc.RpcError as e:
        print(f"✗ Failed to create game: {e.details()}")
        return
    
    # Join game as two players
    players = []
    for i in range(2):
        join_req = game_pb2.JoinGameRequest(
            game_id=game_id,
            player_name=f"TestPlayer{i+1}"
        )
        try:
            join_resp = game_stub.JoinGame(join_req)
            players.append({
                'id': join_resp.player_id,
                'token': join_resp.player_token
            })
            print(f"✓ Player {i+1} joined: ID={join_resp.player_id}")
        except grpc.RpcError as e:
            print(f"✗ Player {i+1} failed to join: {e.details()}")
            return
    
    # Submit a few actions
    print("\nSubmitting test actions...")
    for turn in range(3):
        for player in players:
            action_req = game_pb2.SubmitActionRequest(
                game_id=game_id,
                player_id=player['id'],
                player_token=player['token'],
                action=game_pb2.Action(
                    action_type=game_pb2.ActionType.ACTION_TYPE_WAIT
                )
            )
            try:
                game_stub.SubmitAction(action_req)
                print(f"  Turn {turn+1}: Player {player['id']} submitted action")
            except grpc.RpcError as e:
                print(f"  Turn {turn+1}: Player {player['id']} error: {e.details()}")
    
    # Check experience stats
    print("\nChecking experience stats...")
    stats_req = experience_pb2.GetExperienceStatsRequest(
        game_ids=[game_id]
    )
    
    try:
        stats_resp = exp_stub.GetExperienceStats(stats_req)
        print(f"✓ Experience stats retrieved:")
        print(f"  Total experiences: {stats_resp.total_experiences}")
        print(f"  Total games: {stats_resp.total_games}")
        
        if stats_resp.total_experiences > 0:
            print("\n✅ SUCCESS: Experience collection is working!")
        else:
            print("\n⚠️  WARNING: No experiences collected yet")
            print("    This might be normal if the game just started")
            
    except grpc.RpcError as e:
        print(f"✗ Failed to get stats: {e.details()}")
    
    # Try to stream some experiences
    print("\nAttempting to stream experiences...")
    stream_req = experience_pb2.StreamExperiencesRequest(
        game_ids=[game_id],
        batch_size=5,
        follow=False
    )
    
    try:
        exp_count = 0
        for exp in exp_stub.StreamExperiences(stream_req):
            exp_count += 1
            if exp_count <= 2:
                print(f"  Experience {exp_count}: Turn {exp.turn}, Player {exp.player_id}, Reward {exp.reward}")
            
        print(f"✓ Streamed {exp_count} experiences")
        
        if exp_count > 0:
            print("\n✅ SUCCESS: Experience streaming is working!")
        else:
            print("\n⚠️  No experiences available to stream yet")
            
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            print(f"✗ StreamExperiences not implemented: {e.details()}")
        else:
            print(f"✗ Failed to stream: {e.details()}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()