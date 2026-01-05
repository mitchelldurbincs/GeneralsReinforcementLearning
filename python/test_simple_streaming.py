#!/usr/bin/env python3
"""
Simplified test to check if experience streaming is working.
"""

import sys
import os
import grpc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generals_pb.experience.v1 import experience_pb2, experience_pb2_grpc


def test_experience_stats():
    """Test getting experience statistics."""
    channel = grpc.insecure_channel("localhost:50051")
    exp_stub = experience_pb2_grpc.ExperienceServiceStub(channel)
    
    # Get stats for all games
    request = experience_pb2.GetExperienceStatsRequest(game_ids=[])
    
    try:
        response = exp_stub.GetExperienceStats(request)
        print(f"Total experiences collected: {response.total_experiences}")
        print(f"Total games: {response.total_games}")
        
        if response.total_experiences > 0:
            print(f"Average reward: {response.average_reward:.4f}")
            print(f"Experiences per game:")
            for game_id, count in response.experiences_per_game.items():
                print(f"  {game_id}: {count}")
            return True
        else:
            print("No experiences collected yet")
            return False
            
    except grpc.RpcError as e:
        print(f"Error: {e.code()}: {e.details()}")
        return False


if __name__ == "__main__":
    success = test_experience_stats()
    sys.exit(0 if success else 1)