#!/usr/bin/env python3
"""
Experience Stream Client for Generals Reinforcement Learning

This client connects to the gRPC experience service and streams batched experiences
for reinforcement learning training.
"""

import grpc
import time
import numpy as np
from typing import Iterator, List, Optional, Dict, Any
import logging
from dataclasses import dataclass
from queue import Queue, Empty
import threading

# Import generated protobuf modules
import sys
import os
sys.path.append(os.path.dirname(__file__))

from generals_pb.experience.v1 import experience_pb2
from generals_pb.experience.v1 import experience_pb2_grpc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperienceConfig:
    """Configuration for experience streaming"""
    server_address: str = "localhost:50051"
    game_ids: List[str] = None
    player_ids: List[int] = None
    batch_size: int = 32
    follow: bool = True
    enable_compression: bool = False
    max_batch_wait_ms: int = 100
    buffer_size: int = 1000


class ExperienceStreamClient:
    """Client for streaming experiences from the game server"""
    
    def __init__(self, config: ExperienceConfig):
        self.config = config
        self.channel = None
        self.stub = None
        self.experience_queue = Queue(maxsize=config.buffer_size)
        self.streaming_thread = None
        self.stop_event = threading.Event()
        self.stats = {
            'total_experiences': 0,
            'total_batches': 0,
            'dropped_experiences': 0,
            'last_batch_time': None
        }
        
    def connect(self):
        """Connect to the gRPC server"""
        self.channel = grpc.insecure_channel(self.config.server_address)
        self.stub = experience_pb2_grpc.ExperienceServiceStub(self.channel)
        logger.info(f"Connected to server at {self.config.server_address}")
        
    def disconnect(self):
        """Disconnect from the server"""
        if self.channel:
            self.channel.close()
            logger.info("Disconnected from server")
            
    def start_streaming(self):
        """Start streaming experiences in a background thread"""
        if self.streaming_thread and self.streaming_thread.is_alive():
            logger.warning("Streaming already started")
            return
            
        self.stop_event.clear()
        self.streaming_thread = threading.Thread(target=self._stream_worker)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        logger.info("Started experience streaming")
        
    def stop_streaming(self):
        """Stop streaming experiences"""
        self.stop_event.set()
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5)
        logger.info("Stopped experience streaming")
        
    def _stream_worker(self):
        """Worker thread for streaming experiences"""
        try:
            request = experience_pb2.StreamExperiencesRequest(
                game_ids=self.config.game_ids or [],
                player_ids=self.config.player_ids or [],
                batch_size=self.config.batch_size,
                follow=self.config.follow,
                enable_compression=self.config.enable_compression,
                max_batch_wait_ms=self.config.max_batch_wait_ms
            )
            
            # Stream batches
            for batch in self.stub.StreamExperienceBatches(request):
                if self.stop_event.is_set():
                    break
                    
                self._process_batch(batch)
                
        except grpc.RpcError as e:
            logger.error(f"Streaming error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in streaming: {e}")
            
    def _process_batch(self, batch: experience_pb2.ExperienceBatch):
        """Process a batch of experiences"""
        self.stats['total_batches'] += 1
        self.stats['last_batch_time'] = time.time()
        
        for exp in batch.experiences:
            # Convert experience to numpy arrays
            processed_exp = self._process_experience(exp)
            
            # Try to add to queue
            try:
                self.experience_queue.put(processed_exp, block=False)
                self.stats['total_experiences'] += 1
            except:
                self.stats['dropped_experiences'] += 1
                
        logger.debug(f"Processed batch {batch.batch_id} with {len(batch.experiences)} experiences")
        
    def _process_experience(self, exp: experience_pb2.Experience) -> Dict[str, Any]:
        """Convert protobuf experience to numpy arrays"""
        # Reshape state tensors
        state_shape = exp.state.shape
        state_data = np.array(exp.state.data, dtype=np.float32)
        state = state_data.reshape(state_shape)
        
        next_state_data = np.array(exp.next_state.data, dtype=np.float32)
        next_state = next_state_data.reshape(state_shape)
        
        # Create action mask if available
        action_mask = np.array(exp.action_mask, dtype=np.bool_) if exp.action_mask else None
        
        return {
            'experience_id': exp.experience_id,
            'game_id': exp.game_id,
            'player_id': exp.player_id,
            'turn': exp.turn,
            'state': state,
            'action': exp.action,
            'reward': exp.reward,
            'next_state': next_state,
            'done': exp.done,
            'action_mask': action_mask
        }
        
    def get_experience(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get a single experience from the queue"""
        try:
            return self.experience_queue.get(timeout=timeout)
        except Empty:
            return None
            
    def get_batch(self, batch_size: int, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Get a batch of experiences"""
        batch = []
        deadline = time.time() + timeout
        
        while len(batch) < batch_size and time.time() < deadline:
            exp = self.get_experience(timeout=0.1)
            if exp:
                batch.append(exp)
                
        return batch
        
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            **self.stats,
            'queue_size': self.experience_queue.qsize(),
            'streaming': self.streaming_thread.is_alive() if self.streaming_thread else False
        }


class ExperienceDataset:
    """PyTorch-compatible dataset for streaming experiences"""
    
    def __init__(self, client: ExperienceStreamClient, buffer_size: int = 10000):
        self.client = client
        self.buffer = []
        self.buffer_size = buffer_size
        
    def fill_buffer(self, min_size: int = 1000):
        """Fill the buffer with experiences"""
        while len(self.buffer) < min_size:
            batch = self.client.get_batch(100, timeout=1.0)
            if not batch:
                break
            self.buffer.extend(batch)
            
        # Trim buffer if too large
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
            
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch from the buffer"""
        if len(self.buffer) < batch_size:
            self.fill_buffer(batch_size)
            
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
            
        # Random sample without replacement
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]


def main():
    """Example usage of the experience stream client"""
    
    # Configure client
    config = ExperienceConfig(
        server_address="localhost:50051",
        batch_size=32,
        follow=True
    )
    
    # Create and connect client
    client = ExperienceStreamClient(config)
    client.connect()
    
    try:
        # Start streaming
        client.start_streaming()
        
        # Create dataset
        dataset = ExperienceDataset(client)
        
        # Training loop example
        for epoch in range(10):
            # Fill buffer
            dataset.fill_buffer(min_size=1000)
            
            # Sample batches for training
            for _ in range(10):
                batch = dataset.sample(32)
                if batch:
                    # Process batch for training
                    states = np.stack([exp['state'] for exp in batch])
                    actions = np.array([exp['action'] for exp in batch])
                    rewards = np.array([exp['reward'] for exp in batch])
                    next_states = np.stack([exp['next_state'] for exp in batch])
                    dones = np.array([exp['done'] for exp in batch])
                    
                    logger.info(f"Training batch: states={states.shape}, rewards_mean={rewards.mean():.2f}")
                    
                    # Here you would run your training step
                    # model.train_step(states, actions, rewards, next_states, dones)
                    
            # Print stats
            stats = client.get_stats()
            logger.info(f"Epoch {epoch}: {stats}")
            
            time.sleep(1)
            
    finally:
        # Clean up
        client.stop_streaming()
        client.disconnect()


if __name__ == "__main__":
    main()