#!/usr/bin/env python3
"""
Example of streaming experiences from the Generals game server for RL training.

This example demonstrates:
1. Connecting to the experience streaming service
2. Consuming experiences in real-time
3. Converting experiences to training data
4. Basic statistics tracking
"""

import sys
import os
import time
import argparse
import numpy as np
from collections import deque

# Add parent directory to path to import generals modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generals_agent.experience_consumer import ExperienceConsumer, AsyncExperienceConsumer, experience_to_tensors


def print_experience_info(exp):
    """Print information about a single experience."""
    print(f"\nExperience from Game {exp.metadata['game_id']}, "
          f"Player {exp.metadata['player_id']}, Turn {exp.metadata['turn']}")
    print(f"  State shape: {exp.state.shape}")
    print(f"  Action: {exp.action}")
    print(f"  Reward: {exp.reward:.3f}")
    print(f"  Done: {exp.done}")
    print(f"  Valid actions: {np.sum(exp.action_mask)}/{len(exp.action_mask)}")


def synchronous_streaming_example(server_address: str, num_experiences: int = 10):
    """Example of synchronous experience streaming."""
    print(f"\n=== Synchronous Streaming Example ===")
    print(f"Connecting to {server_address}...")
    
    with ExperienceConsumer(server_address) as consumer:
        # Get initial stats
        stats = consumer.get_experience_stats()
        print(f"\nInitial stats:")
        print(f"  Total experiences: {stats.total_experiences}")
        print(f"  Total games: {stats.total_games}")
        
        # Stream experiences
        print(f"\nStreaming {num_experiences} experiences...")
        experiences_received = 0
        
        try:
            for batch in consumer.stream_experiences(
                batch_size=5,
                follow=False  # Don't wait for new experiences
            ):
                for exp in batch:
                    print_experience_info(exp)
                    experiences_received += 1
                    
                    if experiences_received >= num_experiences:
                        return
                        
        except Exception as e:
            print(f"Error during streaming: {e}")
        
        print(f"\nReceived {experiences_received} experiences")


def asynchronous_streaming_example(server_address: str, duration: int = 30):
    """Example of asynchronous experience streaming with background thread."""
    print(f"\n=== Asynchronous Streaming Example ===")
    print(f"Connecting to {server_address}...")
    
    with AsyncExperienceConsumer(server_address, buffer_size=5000) as consumer:
        # Start streaming in background
        consumer.start_streaming(batch_size=32)
        
        print(f"Streaming in background for {duration} seconds...")
        print("Buffer size will be printed every 5 seconds")
        
        # Collect statistics
        total_experiences = 0
        reward_stats = deque(maxlen=1000)
        
        start_time = time.time()
        last_print_time = start_time
        
        while time.time() - start_time < duration:
            # Get experiences from buffer
            experiences = consumer.get_experiences(batch_size=64, timeout=1.0)
            
            if experiences:
                total_experiences += len(experiences)
                
                # Track rewards
                for exp in experiences:
                    reward_stats.append(exp.reward)
                
                # Optionally convert to training tensors
                if len(experiences) >= 32:
                    states, actions, rewards, next_states, dones, action_masks = \
                        experience_to_tensors(experiences[:32])
                    
                    # Here you would typically:
                    # 1. Add to replay buffer
                    # 2. Sample from replay buffer
                    # 3. Train your model
            
            # Print statistics every 5 seconds
            if time.time() - last_print_time > 5:
                buffer_size = consumer.buffer_size()
                exp_per_sec = total_experiences / (time.time() - start_time)
                
                print(f"\nStats at {int(time.time() - start_time)}s:")
                print(f"  Total experiences: {total_experiences}")
                print(f"  Experiences/sec: {exp_per_sec:.1f}")
                print(f"  Buffer size: {buffer_size}")
                
                if reward_stats:
                    print(f"  Avg reward: {np.mean(reward_stats):.3f}")
                    print(f"  Min/Max reward: {min(reward_stats):.3f}/{max(reward_stats):.3f}")
                
                last_print_time = time.time()
        
        print(f"\n\nFinal statistics:")
        print(f"  Total experiences collected: {total_experiences}")
        print(f"  Average rate: {total_experiences/duration:.1f} exp/sec")


def training_loop_example(server_address: str):
    """Example of how to integrate experience streaming into a training loop."""
    print(f"\n=== Training Loop Example ===")
    print(f"Connecting to {server_address}...")
    
    # This is a skeleton of what a real training loop might look like
    with AsyncExperienceConsumer(server_address, buffer_size=10000) as consumer:
        # Start streaming experiences
        consumer.start_streaming(batch_size=64)
        
        # Wait for initial experiences
        print("Waiting for initial experiences...")
        while consumer.buffer_size() < 1000:
            time.sleep(0.1)
        
        print("Starting training loop...")
        
        # Training loop
        for step in range(100):
            # Get a batch of experiences
            batch = consumer.get_experiences(batch_size=256, timeout=2.0)
            
            if len(batch) < 32:
                print(f"Step {step}: Not enough experiences ({len(batch)}), skipping...")
                continue
            
            # Convert to tensors
            states, actions, rewards, next_states, dones, action_masks = \
                experience_to_tensors(batch)
            
            # Simulate training step
            loss = np.random.random()  # In reality, this would be your model loss
            
            if step % 10 == 0:
                print(f"Step {step}: Batch size={len(batch)}, "
                      f"Avg reward={np.mean(rewards):.3f}, Loss={loss:.4f}")
            
            # Simulate training time
            time.sleep(0.1)
        
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Stream experiences from Generals game server"
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)"
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "async", "train", "all"],
        default="all",
        help="Which example to run"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration for async example in seconds"
    )
    parser.add_argument(
        "--num-experiences",
        type=int,
        default=10,
        help="Number of experiences for sync example"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode in ["sync", "all"]:
            synchronous_streaming_example(args.server, args.num_experiences)
        
        if args.mode in ["async", "all"]:
            asynchronous_streaming_example(args.server, args.duration)
        
        if args.mode in ["train", "all"]:
            training_loop_example(args.server)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()