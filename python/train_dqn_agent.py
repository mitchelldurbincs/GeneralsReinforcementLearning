#!/usr/bin/env python3
"""
Basic DQN training example for Generals.io using the Gym environment.

This demonstrates how to use the GeneralsEnv with PyTorch to train
a Deep Q-Network agent for playing Generals.io.
"""

import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, List
import gymnasium as gym

# Add parent directory to path for imports
sys.path.insert(0, '/home/aspect/source/GeneralsReinforcementLearning/python')

from generals_gym import GeneralsEnv


class DQN(nn.Module):
    """Deep Q-Network for Generals.io."""
    
    def __init__(self, input_channels: int, board_size: int, action_size: int):
        """
        Initialize the DQN.
        
        Args:
            input_channels: Number of input channels (9 for our observation)
            board_size: Size of the board (width * height)
            action_size: Number of possible actions
        """
        super(DQN, self).__init__()
        
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate the size after convolutions (same size due to padding)
        # For a 10x10 board, this would be 128 * 10 * 10 = 12800
        conv_output_size = 128 * board_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int):
        """Initialize replay buffer with given capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences from the buffer."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for playing Generals.io."""
    
    def __init__(
        self,
        env: GeneralsEnv,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100
    ):
        """
        Initialize DQN agent.
        
        Args:
            env: Generals Gym environment
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Get environment parameters
        obs_shape = env.observation_space.shape
        self.input_channels = obs_shape[0]
        self.board_height = obs_shape[1]
        self.board_width = obs_shape[2]
        self.board_size = self.board_height * self.board_width
        self.action_size = env.action_space.n
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQN(self.input_channels, self.board_size, self.action_size).to(self.device)
        self.target_network = DQN(self.input_channels, self.board_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.steps_done = 0
        self.episodes_done = 0
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def select_action(self, state: np.ndarray, valid_actions_mask: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy with action masking.
        
        Args:
            state: Current state observation
            valid_actions_mask: Boolean mask of valid actions
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random valid action
            valid_indices = np.where(valid_actions_mask)[0]
            if len(valid_indices) > 0:
                return np.random.choice(valid_indices)
            else:
                return 0  # Default action if no valid actions
        else:
            # Greedy action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
                
                # Mask invalid actions
                q_values[~valid_actions_mask] = -float('inf')
                
                return int(np.argmax(q_values))
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, num_episodes: int = 1000):
        """
        Train the DQN agent.
        
        Args:
            num_episodes: Number of episodes to train for
        """
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        for episode in range(num_episodes):
            # Reset environment
            state, info = self.env.reset()
            valid_actions_mask = info.get('valid_actions_mask', np.ones(self.action_size, dtype=bool))
            
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action
                action = self.select_action(state, valid_actions_mask)
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_valid_actions_mask = info.get('valid_actions_mask', np.ones(self.action_size, dtype=bool))
                
                # Store transition
                done = terminated or truncated
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                valid_actions_mask = next_valid_actions_mask
                episode_reward += reward
                episode_length += 1
                
                # Train
                loss = self.train_step()
                if loss is not None:
                    losses.append(loss)
                
                # Update target network
                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()
                
                self.steps_done += 1
                
                if done:
                    break
            
            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            self.episodes_done += 1
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                
                self.logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.1f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )
        
        return episode_rewards, episode_lengths
    
    def save_model(self, path: str):
        """Save the model weights."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.logger.info(f"Model loaded from {path}")


def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("DQN Training for Generals.io")
    print("=" * 60)
    
    # Create environment
    print("\nCreating environment...")
    env = GeneralsEnv(
        server_address="localhost:50051",
        board_width=10,
        board_height=10,
        max_players=2,
        fog_of_war=True,
        max_turns=200
    )
    print("✓ Environment created")
    
    # Create agent
    print("\nCreating DQN agent...")
    agent = DQNAgent(
        env,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32,
        target_update_freq=100
    )
    print(f"✓ Agent created (Device: {agent.device})")
    
    # Train agent
    print("\nStarting training...")
    print(f"Training for 100 episodes...")
    episode_rewards, episode_lengths = agent.train(num_episodes=100)
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Average reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Average length (last 10 episodes): {np.mean(episode_lengths[-10:]):.1f}")
    print(f"Total steps: {agent.steps_done}")
    
    # Save model
    model_path = "dqn_generals_model.pth"
    agent.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Close environment
    env.close()
    print("\n✓ Training complete and environment closed")


if __name__ == "__main__":
    import grpc
    from generals_pb.game.v1 import game_pb2, game_pb2_grpc
    
    # Check if server is running
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(width=5, height=5, max_players=2)
        ))
        channel.close()
        
        print("✓ Game server is running")
        main()
        
    except grpc.RpcError:
        print("\n✗ Error: Game server is not running!")
        print("  Please start the server with: go run cmd/game_server/main.go")
        sys.exit(1)