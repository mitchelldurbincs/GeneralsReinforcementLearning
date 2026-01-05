#!/usr/bin/env python3
"""Simple test client to verify streaming works"""

import grpc
import sys
import os
import time

# Add path to generated proto files
sys.path.append(os.path.dirname(__file__))

from generals_pb.experience.v1 import experience_pb2
from generals_pb.experience.v1 import experience_pb2_grpc

def test_streaming():
    # Connect to server
    channel = grpc.insecure_channel('localhost:50051')
    stub = experience_pb2_grpc.ExperienceServiceStub(channel)
    
    print("Connected to server, starting stream...")
    
    # Create request
    request = experience_pb2.StreamExperiencesRequest(
        game_ids=["test-game-1"],
        batch_size=10,
        follow=True
    )
    
    # Stream experiences (using the old single-experience method first to test)
    count = 0
    start_time = time.time()
    
    try:
        for experience in stub.StreamExperiences(request):
            count += 1
            if count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Received {count} experiences in {elapsed:.2f}s - Latest: game={experience.game_id}, player={experience.player_id}, reward={experience.reward:.2f}")
            
            if count >= 50:
                print(f"Successfully received {count} experiences!")
                break
                
    except grpc.RpcError as e:
        print(f"RPC Error: {e}")
        return False
    
    return True

def test_batch_streaming():
    # Connect to server
    channel = grpc.insecure_channel('localhost:50051')
    stub = experience_pb2_grpc.ExperienceServiceStub(channel)
    
    print("\nTesting batch streaming...")
    
    # Create request
    request = experience_pb2.StreamExperiencesRequest(
        game_ids=["test-game-1"],
        batch_size=10,
        follow=True
    )
    
    batch_count = 0
    exp_count = 0
    
    try:
        for batch in stub.StreamExperienceBatches(request):
            batch_count += 1
            exp_count += len(batch.experiences)
            print(f"Received batch {batch.batch_id} with {len(batch.experiences)} experiences (total: {exp_count})")
            
            if batch_count >= 5:
                print(f"Successfully received {batch_count} batches with {exp_count} total experiences!")
                break
                
    except grpc.RpcError as e:
        print(f"RPC Error in batch streaming: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing experience streaming...")
    
    # Test single experience streaming
    if test_streaming():
        print("✓ Single experience streaming works!")
    else:
        print("✗ Single experience streaming failed!")
    
    # Test batch streaming
    if test_batch_streaming():
        print("✓ Batch streaming works!")
    else:
        print("✗ Batch streaming failed!")