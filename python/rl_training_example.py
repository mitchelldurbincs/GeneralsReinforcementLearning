#!/usr/bin/env python3
"""
Reinforcement Learning Training Example

This example shows how to integrate the experience streaming with a simple DQN agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
import logging
from experience_stream_client import ExperienceStreamClient, ExperienceConfig, ExperienceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Simple DQN network for Generals.io"""
    
    def __init__(self, input_channels: int = 9, board_size: int = 20, num_actions: int = None):
        super().__init__()
        
        # Calculate number of actions (4 directions * board_size^2)
        self.num_actions = num_actions or (4 * board_size * board_size)
        
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = board_size * board_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_actions)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class DQNAgent:
    """DQN Agent for training on Generals.io experiences"""
    
    def __init__(self, 
                 input_channels: int = 9,
                 board_size: int = 20,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.board_size = board_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Create Q-network and target network
        self.q_network = DQNNetwork(input_channels, board_size).to(self.device)
        self.target_network = DQNNetwork(input_channels, board_size).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Training stats
        self.training_step = 0
        self.losses = []
        
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def train_batch(self, experiences: List[Dict]) -> float:
        """Train on a batch of experiences"""
        # Convert experiences to tensors
        states = torch.FloatTensor([exp['state'] for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update training step
        self.training_step += 1
        
        # Store loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
        
    def save(self, path: str):
        """Save the model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        logger.info(f"Model loaded from {path}")


def train_agent(config: ExperienceConfig, 
                num_epochs: int = 100,
                batches_per_epoch: int = 50,
                batch_size: int = 32,
                update_target_every: int = 1000,
                save_every: int = 10):
    """Main training loop"""
    
    # Create experience client
    client = ExperienceStreamClient(config)
    client.connect()
    client.start_streaming()
    
    # Create dataset
    dataset = ExperienceDataset(client, buffer_size=10000)
    
    # Create agent
    agent = DQNAgent(
        input_channels=9,  # 9 channels as defined in the proto
        board_size=20,
        learning_rate=0.001,
        gamma=0.99
    )
    
    try:
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Fill buffer with new experiences
            logger.info(f"Epoch {epoch}: Filling buffer...")
            dataset.fill_buffer(min_size=batch_size * batches_per_epoch)
            
            # Train on batches
            for batch_idx in range(batches_per_epoch):
                # Sample batch
                batch = dataset.sample(batch_size)
                
                if len(batch) < batch_size // 2:
                    logger.warning(f"Not enough experiences in buffer: {len(batch)}")
                    continue
                    
                # Train on batch
                loss = agent.train_batch(batch)
                epoch_losses.append(loss)
                
                # Update target network periodically
                if agent.training_step % update_target_every == 0:
                    agent.update_target_network()
                    logger.info(f"Updated target network at step {agent.training_step}")
                    
            # Log epoch stats
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                logger.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f}, epsilon={agent.epsilon:.4f}, "
                          f"buffer_size={len(dataset.buffer)}, training_step={agent.training_step}")
            
            # Get streaming stats
            stream_stats = client.get_stats()
            logger.info(f"Stream stats: {stream_stats}")
            
            # Save model periodically
            if (epoch + 1) % save_every == 0:
                agent.save(f"models/dqn_epoch_{epoch+1}.pt")
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    finally:
        # Clean up
        client.stop_streaming()
        client.disconnect()
        
        # Save final model
        agent.save("models/dqn_final.pt")
        
    return agent


def main():
    """Example training run"""
    
    # Configure experience streaming
    config = ExperienceConfig(
        server_address="localhost:50051",
        batch_size=64,
        follow=True,
        buffer_size=10000
    )
    
    # Train agent
    agent = train_agent(
        config=config,
        num_epochs=100,
        batches_per_epoch=50,
        batch_size=32,
        update_target_every=1000,
        save_every=10
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()