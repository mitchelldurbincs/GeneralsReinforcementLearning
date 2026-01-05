#!/usr/bin/env python3
"""
Test script for the Generals Gym environment.
Runs a random agent in the environment to verify it works correctly.
"""

import sys
import logging
import numpy as np
import gymnasium as gym

# Add parent directory to path for imports
sys.path.insert(0, '/home/aspect/source/GeneralsReinforcementLearning/python')

from generals_gym import GeneralsEnv


def test_gym_environment():
    """Test the Gym environment with random actions."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Generals Gym Environment")
    print("=" * 50)
    
    try:
        # Create environment
        print("\n1. Creating environment...")
        env = GeneralsEnv(
            server_address="localhost:50051",
            board_width=10,  # Smaller board for testing
            board_height=10,
            max_players=2,
            fog_of_war=True,
            render_mode="human",
            max_turns=100
        )
        print("✓ Environment created successfully")
        
        # Test reset
        print("\n2. Testing reset...")
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation dtype: {obs.dtype}")
        print(f"  - Game ID: {info['game_id']}")
        print(f"  - Player ID: {info['player_id']}")
        
        # Test action space
        print("\n3. Testing action space...")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Number of actions: {env.action_space.n}")
        
        # Test observation space
        print("\n4. Testing observation space...")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Observation shape: {env.observation_space.shape}")
        
        # Run a few steps with random valid actions
        print("\n5. Running random agent for 20 steps...")
        total_reward = 0
        step_count = 0
        
        for step in range(20):
            # Get valid actions mask
            valid_actions = info.get('valid_actions_mask', np.ones(env.action_space.n))
            valid_indices = np.where(valid_actions)[0]
            
            if len(valid_indices) > 0:
                # Select random valid action
                action = np.random.choice(valid_indices)
            else:
                # No valid actions, pass
                action = 0
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            print(f"  Step {step + 1}: Action={action}, Reward={reward:.2f}, "
                  f"Turn={info.get('turn', 0)}, Status={info.get('game_status', 'UNKNOWN')}")
            
            if terminated or truncated:
                print(f"\n  Episode ended after {step_count} steps")
                print(f"  Total reward: {total_reward:.2f}")
                if 'winner' in info:
                    print(f"  Winner: Player {info['winner']}")
                break
        
        # Test multiple episodes
        print("\n6. Testing multiple episodes...")
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while episode_length < 50:  # Limit steps per episode
                valid_actions = info.get('valid_actions_mask', np.ones(env.action_space.n))
                valid_indices = np.where(valid_actions)[0]
                
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                else:
                    action = 0
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Episode {episode + 1}: Length={episode_length}, Reward={episode_reward:.2f}")
        
        print(f"\n✓ All tests passed!")
        print(f"  Average episode length: {np.mean(episode_lengths):.1f}")
        print(f"  Average episode reward: {np.mean(episode_rewards):.2f}")
        
        # Close environment
        env.close()
        print("\n✓ Environment closed successfully")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_observation_channels():
    """Test that observation channels are correctly formatted."""
    print("\n" + "=" * 50)
    print("Testing Observation Channels")
    print("=" * 50)
    
    try:
        env = GeneralsEnv(
            server_address="localhost:50051",
            board_width=5,  # Very small for easy inspection
            board_height=5,
            max_players=2
        )
        
        obs, info = env.reset()
        
        print("\nObservation channel analysis:")
        channel_names = [
            "Visibility", "Ownership", "Army Count", "Empty Tiles",
            "Mountains", "Cities", "Generals", "Turn Counter", "Valid Actions"
        ]
        
        for i, name in enumerate(channel_names[:obs.shape[0]]):
            channel = obs[i]
            print(f"\nChannel {i} - {name}:")
            print(f"  Min: {channel.min():.3f}, Max: {channel.max():.3f}")
            print(f"  Mean: {channel.mean():.3f}, Std: {channel.std():.3f}")
            print(f"  Non-zero count: {np.count_nonzero(channel)}")
        
        env.close()
        print("\n✓ Observation channels test complete")
        
    except Exception as e:
        print(f"\n✗ Observation test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Generals Gym Environment Test Suite")
    print("=" * 50)
    
    # Check if server is running
    import grpc
    from generals_pb.game.v1 import game_pb2, game_pb2_grpc
    
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(width=5, height=5, max_players=2)
        ))
        channel.close()
        
        print("✓ Game server is running")
        
        # Run tests
        success = test_gym_environment()
        if success:
            test_observation_channels()
        
        sys.exit(0 if success else 1)
        
    except grpc.RpcError:
        print("\n✗ Error: Game server is not running!")
        print("  Please start the server with: go run cmd/game_server/main.go")
        sys.exit(1)