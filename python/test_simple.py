#!/usr/bin/env python3
"""Quick test to see if experiences are being received"""

import grpc
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
from generals_pb.experience.v1 import experience_pb2, experience_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = experience_pb2_grpc.ExperienceServiceStub(channel)

print("Connected, requesting stream...")
request = experience_pb2.StreamExperiencesRequest(
    batch_size=10,
    follow=False  # Don't follow, just get existing
)

print("Starting stream...")
count = 0
try:
    for experience in stub.StreamExperiences(request, timeout=5):
        count += 1
        print(f"Got experience {count}: game={experience.game_id}, reward={experience.reward}")
        if count >= 10:
            break
except grpc.RpcError as e:
    print(f"Error: {e}")

print(f"Total experiences received: {count}")