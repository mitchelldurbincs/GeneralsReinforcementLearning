#!/usr/bin/env python3
"""Minimal test of the Gym environment to isolate issues."""

import sys
sys.path.insert(0, '/home/aspect/source/GeneralsReinforcementLearning/python')

from generals_gym import GeneralsEnv
import numpy as np

print("Testing minimal Gym environment...")

# Create env
env = GeneralsEnv(
    server_address="localhost:50051",
    board_width=5,
    board_height=5,
    max_players=2,
    fog_of_war=False,
    max_turns=10
)

print("Environment created")

# Reset
obs, info = env.reset()
print(f"Reset successful, game_id: {info['game_id']}")
print(f"Observation shape: {obs.shape}")
print(f"Valid actions: {np.sum(info['valid_actions_mask'])}")

# Take a few steps
for i in range(5):
    # Pick random valid action
    valid_mask = info.get('valid_actions_mask', np.ones(env.action_space.n))
    valid_actions = np.where(valid_mask)[0]
    
    if len(valid_actions) > 0:
        action = np.random.choice(valid_actions)
        print(f"\nStep {i+1}: Taking action {action}")
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Reward: {reward:.2f}")
            print(f"  Turn: {info.get('turn', 0)}")
            print(f"  Done: {terminated or truncated}")
            
            if terminated or truncated:
                print(f"  Game ended! Winner: {info.get('winner', 'unknown')}")
                break
        except Exception as e:
            print(f"  Error during step: {e}")
            break
    else:
        print(f"\nStep {i+1}: No valid actions available")
        break

env.close()
print("\nTest complete!")