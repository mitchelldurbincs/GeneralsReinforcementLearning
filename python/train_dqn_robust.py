#!/usr/bin/env python3
"""
Robust DQN training for Generals.io with error recovery and checkpointing.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import logging
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/aspect/source/GeneralsReinforcementLearning/python')
from generals_gym import GeneralsEnv


class SimpleDQN(nn.Module):
    """Simple but effective DQN network."""
    
    def __init__(self, input_shape, n_actions):
        super(SimpleDQN, self).__init__()
        channels, height, width = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate flattened size
        conv_out_size = 64 * height * width
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, n_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RobustDQNTrainer:
    """Robust DQN trainer with error recovery and checkpointing."""
    
    def __init__(self, config):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = -float('inf')
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Initialize environment
        self.env = None
        self.reset_environment()
        
        # Initialize networks
        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n
        
        self.q_network = SimpleDQN(obs_shape, n_actions).to(self.device)
        self.target_network = SimpleDQN(obs_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        
        # Experience replay
        self.memory = deque(maxlen=config['buffer_size'])
        
        # Exploration
        self.epsilon = config['epsilon_start']
        
        # Create checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def reset_environment(self):
        """Reset or create environment with error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.env:
                    self.env.close()
                
                self.env = GeneralsEnv(
                    server_address=self.config['server_address'],
                    board_width=self.config['board_width'],
                    board_height=self.config['board_height'],
                    max_players=2,
                    fog_of_war=self.config.get('fog_of_war', True),
                    max_turns=self.config.get('max_turns', 200)
                )
                
                self.logger.info("Environment (re)created successfully")
                return
                
            except Exception as e:
                self.logger.warning(f"Environment creation attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        
        raise RuntimeError("Failed to create environment after multiple attempts")
    
    def select_action(self, state, valid_mask):
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Random valid action
            valid_actions = np.where(valid_mask)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions)
            return 0
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
                q_values[~valid_mask] = -float('inf')
                return int(np.argmax(q_values))
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + self.config['gamma'] * next_q * (1 - dones)
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def run_episode(self):
        """Run a single episode with error recovery."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Reset environment
                state, info = self.env.reset()
                valid_mask = info.get('valid_actions_mask', np.ones(self.env.action_space.n, dtype=bool))
                
                episode_reward = 0
                episode_length = 0
                
                while episode_length < self.config.get('max_steps_per_episode', 1000):
                    # Select and execute action
                    action = self.select_action(state, valid_mask)
                    next_state, reward, terminated, truncated, next_info = self.env.step(action)
                    
                    done = terminated or truncated
                    valid_mask = next_info.get('valid_actions_mask', np.ones(self.env.action_space.n, dtype=bool))
                    
                    # Store experience
                    self.memory.append((state, action, reward, next_state, done))
                    
                    # Train
                    loss = self.train_step()
                    if loss is not None:
                        self.losses.append(loss)
                    
                    # Update state
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    self.total_steps += 1
                    
                    # Update target network
                    if self.total_steps % self.config['target_update_freq'] == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
                    
                    if done:
                        break
                
                # Episode completed successfully
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.total_episodes += 1
                
                # Update epsilon
                self.epsilon = max(
                    self.config['epsilon_end'],
                    self.epsilon * self.config['epsilon_decay']
                )
                
                return episode_reward, episode_length
                
            except Exception as e:
                self.logger.warning(f"Episode attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    self.reset_environment()
                else:
                    raise
        
        return 0, 0  # Failed episode
    
    def save_checkpoint(self, filename=None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_ep{self.total_episodes}.pth"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'episode': self.total_episodes,
            'total_steps': self.total_steps,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_reward': self.best_reward,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
        
        # Save metrics
        metrics_file = os.path.join(self.checkpoint_dir, 'training_metrics.json')
        metrics = {
            'episode_rewards': self.episode_rewards[-100:],  # Last 100
            'episode_lengths': self.episode_lengths[-100:],
            'avg_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath)
        
        self.total_episodes = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def train(self, num_episodes):
        """Main training loop with error recovery."""
        self.logger.info(f"Starting training for {num_episodes} episodes")
        self.logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            try:
                # Run episode
                episode_reward, episode_length = self.run_episode()
                
                # Log progress
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
                    avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                    
                    elapsed = time.time() - start_time
                    eps_per_sec = (episode + 1) / elapsed
                    
                    self.logger.info(
                        f"Episode {episode + 1}/{num_episodes} | "
                        f"Reward: {avg_reward:.2f} | Length: {avg_length:.1f} | "
                        f"Loss: {avg_loss:.4f} | Epsilon: {self.epsilon:.3f} | "
                        f"Speed: {eps_per_sec:.2f} eps/s"
                    )
                    
                    # Save best model
                    if avg_reward > self.best_reward:
                        self.best_reward = avg_reward
                        self.save_checkpoint('best_model.pth')
                
                # Periodic checkpoint
                if (episode + 1) % 50 == 0:
                    self.save_checkpoint()
                
            except KeyboardInterrupt:
                self.logger.info("Training interrupted by user")
                self.save_checkpoint('interrupted.pth')
                break
            except Exception as e:
                self.logger.error(f"Training error in episode {episode + 1}: {e}")
                time.sleep(2)
                self.reset_environment()
        
        # Final save
        self.save_checkpoint('final_model.pth')
        
        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed:.1f} seconds")
        self.logger.info(f"Final stats - Episodes: {self.total_episodes}, Steps: {self.total_steps}")
        
        if self.env:
            self.env.close()


def main():
    """Main training function."""
    
    # Training configuration
    config = {
        'server_address': 'localhost:50051',
        'board_width': 10,
        'board_height': 10,
        'fog_of_war': True,
        'max_turns': 200,
        'max_steps_per_episode': 200,
        
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 5000,
        'batch_size': 32,
        'target_update_freq': 100,
        
        'checkpoint_dir': 'dqn_checkpoints'
    }
    
    print("=" * 60)
    print("Robust DQN Training for Generals.io")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create trainer
    trainer = RobustDQNTrainer(config)
    
    # Train
    trainer.train(num_episodes=50)
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    import grpc
    from generals_pb.game.v1 import game_pb2, game_pb2_grpc
    
    # Check server
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(width=5, height=5, max_players=2)
        ))
        channel.close()
        
        print("✓ Game server is running\n")
        main()
        
    except grpc.RpcError as e:
        print(f"\n✗ Error: Game server is not running!")
        print(f"  Details: {e}")
        print("  Please start the server with: go run cmd/game_server/main.go")
        sys.exit(1)