#!/usr/bin/env python3
"""
Simplified DQN training for Generals.io - focusing on getting it working first.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import logging

sys.path.insert(0, '/home/aspect/source/GeneralsReinforcementLearning/python')
from generals_gym import GeneralsEnv


class SimpleDQN(nn.Module):
    """Simplified DQN network."""
    
    def __init__(self, input_shape, n_actions):
        super(SimpleDQN, self).__init__()
        channels, height, width = input_shape
        
        # Simple convolutional layers
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate flattened size
        conv_out_size = 32 * height * width
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_simple_dqn(episodes=10):
    """Train a simple DQN agent."""
    
    print("=" * 60)
    print("Simple DQN Training for Generals.io")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = GeneralsEnv(
        server_address="localhost:50051",
        board_width=5,  # Small board for faster training
        board_height=5,
        max_players=2,
        fog_of_war=False,  # No fog for simpler learning
        max_turns=100
    )
    print("✓ Environment created")
    
    # Get environment info
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    print(f"  Observation shape: {obs_shape}")
    print(f"  Number of actions: {n_actions}")
    
    # Create network
    print("\n2. Creating DQN network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = SimpleDQN(obs_shape, n_actions).to(device)
    target_network = SimpleDQN(obs_shape, n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    print(f"✓ Network created (device: {device})")
    
    # Replay buffer
    memory = deque(maxlen=2000)
    
    # Training parameters
    epsilon = 1.0
    epsilon_decay = 0.99
    epsilon_min = 0.1
    gamma = 0.95
    batch_size = 32
    target_update = 10
    
    print(f"\n3. Starting training for {episodes} episodes...")
    print("-" * 60)
    
    all_rewards = []
    all_lengths = []
    
    for episode in range(episodes):
        # Reset environment
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while episode_length < 100:  # Max steps per episode
            # Select action
            if random.random() < epsilon:
                # Random action from valid actions
                valid_mask = info.get('valid_actions_mask', np.ones(n_actions))
                valid_actions = np.where(valid_mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0
            else:
                # Greedy action from network
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_network(state_tensor).cpu().numpy()[0]
                    
                    # Mask invalid actions
                    valid_mask = info.get('valid_actions_mask', np.ones(n_actions))
                    q_values[~valid_mask] = -float('inf')
                    action = int(np.argmax(q_values))
            
            # Take action
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store in memory
            memory.append((state, action, reward, next_state, done))
            
            # Update state
            state = next_state
            info = next_info
            episode_reward += reward
            episode_length += 1
            
            # Train if enough samples
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                current_q = q_network(states).gather(1, actions.unsqueeze(1))
                next_q = target_network(next_states).max(1)[0].detach()
                target_q = rewards + gamma * next_q * (1 - dones)
                
                loss = F.mse_loss(current_q.squeeze(), target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        # Update target network
        if episode % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Record stats
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        
        # Print progress
        print(f"Episode {episode + 1:3d} | Reward: {episode_reward:6.2f} | "
              f"Length: {episode_length:3d} | Epsilon: {epsilon:.3f}")
    
    print("-" * 60)
    print(f"\n✓ Training complete!")
    print(f"  Average reward: {np.mean(all_rewards):.2f}")
    print(f"  Average length: {np.mean(all_lengths):.1f}")
    print(f"  Final epsilon: {epsilon:.3f}")
    
    # Save model
    torch.save(q_network.state_dict(), 'simple_dqn_model.pth')
    print(f"\n✓ Model saved to simple_dqn_model.pth")
    
    env.close()
    return all_rewards, all_lengths


if __name__ == "__main__":
    # Check server connection
    import grpc
    from generals_pb.game.v1 import game_pb2, game_pb2_grpc
    
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(width=5, height=5, max_players=2)
        ))
        channel.close()
        print("✓ Game server is running\n")
        
        # Run training
        rewards, lengths = train_simple_dqn(episodes=20)
        
    except grpc.RpcError as e:
        print(f"\n✗ Error: Game server is not running!")
        print(f"  Error details: {e}")
        print("  Please start the server with: go run cmd/game_server/main.go")
        sys.exit(1)