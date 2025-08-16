"""Experience consumer for streaming RL training experiences from the game server."""

import grpc
from typing import Iterator, Optional, List, Dict, Any, NamedTuple
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass

from generals_pb.experience.v1 import experience_pb2, experience_pb2_grpc


@dataclass
class Experience:
    """Container for a single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_mask: np.ndarray
    metadata: Dict[str, Any]


class ExperienceConsumer:
    """Consumes experiences from the Go game server for training."""
    
    def __init__(self, server_address: str = "localhost:50051"):
        """Initialize the experience consumer.
        
        Args:
            server_address: The gRPC server address
        """
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self._connect()
    
    def _connect(self):
        """Establish connection to the gRPC server."""
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = experience_pb2_grpc.ExperienceServiceStub(self.channel)
    
    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def stream_experiences(
        self,
        game_ids: Optional[List[str]] = None,
        player_ids: Optional[List[int]] = None,
        batch_size: int = 32,
        follow: bool = True,
        min_turn: int = 0
    ) -> Iterator[List[Experience]]:
        """Stream experiences from the server.
        
        Args:
            game_ids: List of game IDs to filter by (None = all games)
            player_ids: List of player IDs to filter by (None = all players)
            batch_size: Number of experiences to batch together
            follow: Whether to keep streaming new experiences
            min_turn: Minimum turn number to include
            
        Yields:
            Batches of Experience objects
        """
        request = experience_pb2.StreamExperiencesRequest(
            game_ids=game_ids or [],
            player_ids=player_ids or [],
            batch_size=batch_size,
            follow=follow,
            min_turn=min_turn
        )
        
        batch = []
        try:
            for exp_proto in self.stub.StreamExperiences(request):
                # Convert protobuf to Experience object
                experience = self._proto_to_experience(exp_proto)
                batch.append(experience)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            
            # Yield any remaining experiences
            if batch:
                yield batch
                
        except grpc.RpcError as e:
            print(f"gRPC error in stream_experiences: {e.code()}: {e.details()}")
            raise
    
    def submit_experiences(
        self,
        experiences: List[experience_pb2.Experience]
    ) -> experience_pb2.SubmitExperiencesResponse:
        """Submit experiences to the server (for distributed collection).
        
        Args:
            experiences: List of experience protobuf messages
            
        Returns:
            Response with acceptance/rejection counts
        """
        request = experience_pb2.SubmitExperiencesRequest(
            experiences=experiences
        )
        
        try:
            return self.stub.SubmitExperiences(request)
        except grpc.RpcError as e:
            print(f"gRPC error in submit_experiences: {e.code()}: {e.details()}")
            raise
    
    def get_experience_stats(
        self,
        game_ids: Optional[List[str]] = None
    ) -> experience_pb2.GetExperienceStatsResponse:
        """Get statistics about collected experiences.
        
        Args:
            game_ids: List of game IDs to get stats for (None = all games)
            
        Returns:
            Statistics about collected experiences
        """
        request = experience_pb2.GetExperienceStatsRequest(
            game_ids=game_ids or []
        )
        
        try:
            return self.stub.GetExperienceStats(request)
        except grpc.RpcError as e:
            print(f"gRPC error in get_experience_stats: {e.code()}: {e.details()}")
            raise
    
    def _proto_to_experience(self, exp_proto: experience_pb2.Experience) -> Experience:
        """Convert protobuf experience to Experience object.
        
        Args:
            exp_proto: Protobuf experience message
            
        Returns:
            Experience object with numpy arrays
        """
        # Convert state tensor
        state_shape = exp_proto.state.shape
        state = np.array(exp_proto.state.data, dtype=np.float32).reshape(state_shape)
        
        # Convert next state tensor
        next_state_shape = exp_proto.next_state.shape
        next_state = np.array(exp_proto.next_state.data, dtype=np.float32).reshape(next_state_shape)
        
        # Convert action mask
        action_mask = np.array(exp_proto.action_mask, dtype=np.bool_)
        
        # Extract metadata
        metadata = {
            'experience_id': exp_proto.experience_id,
            'game_id': exp_proto.game_id,
            'player_id': exp_proto.player_id,
            'turn': exp_proto.turn,
            'collected_at': exp_proto.collected_at.ToDatetime() if exp_proto.collected_at else None,
            **dict(exp_proto.metadata)
        }
        
        return Experience(
            state=state,
            action=exp_proto.action,
            reward=exp_proto.reward,
            next_state=next_state,
            done=exp_proto.done,
            action_mask=action_mask,
            metadata=metadata
        )


class AsyncExperienceConsumer:
    """Asynchronous experience consumer that streams in the background."""
    
    def __init__(
        self,
        server_address: str = "localhost:50051",
        buffer_size: int = 10000
    ):
        """Initialize the async experience consumer.
        
        Args:
            server_address: The gRPC server address
            buffer_size: Size of the internal experience buffer
        """
        self.consumer = ExperienceConsumer(server_address)
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.streaming_thread = None
        self.stop_event = threading.Event()
    
    def start_streaming(
        self,
        game_ids: Optional[List[str]] = None,
        player_ids: Optional[List[int]] = None,
        batch_size: int = 32
    ):
        """Start streaming experiences in the background.
        
        Args:
            game_ids: List of game IDs to filter by
            player_ids: List of player IDs to filter by
            batch_size: Batch size for streaming
        """
        if self.streaming_thread and self.streaming_thread.is_alive():
            print("Streaming already in progress")
            return
        
        self.stop_event.clear()
        self.streaming_thread = threading.Thread(
            target=self._stream_worker,
            args=(game_ids, player_ids, batch_size),
            daemon=True
        )
        self.streaming_thread.start()
    
    def stop_streaming(self):
        """Stop the background streaming thread."""
        self.stop_event.set()
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5.0)
    
    def get_experiences(self, batch_size: int, timeout: float = 1.0) -> List[Experience]:
        """Get a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to retrieve
            timeout: Timeout in seconds to wait for experiences
            
        Returns:
            List of Experience objects (may be less than batch_size if timeout)
        """
        batch = []
        deadline = time.time() + timeout
        
        while len(batch) < batch_size and time.time() < deadline:
            try:
                remaining_timeout = deadline - time.time()
                if remaining_timeout <= 0:
                    break
                    
                exp = self.buffer.get(timeout=remaining_timeout)
                batch.append(exp)
            except queue.Empty:
                break
        
        return batch
    
    def buffer_size(self) -> int:
        """Get the current buffer size."""
        return self.buffer.qsize()
    
    def _stream_worker(
        self,
        game_ids: Optional[List[str]],
        player_ids: Optional[List[int]],
        batch_size: int
    ):
        """Worker thread that streams experiences."""
        print(f"Starting experience streaming from {self.consumer.server_address}")
        
        try:
            for batch in self.consumer.stream_experiences(
                game_ids=game_ids,
                player_ids=player_ids,
                batch_size=batch_size,
                follow=True
            ):
                if self.stop_event.is_set():
                    break
                
                # Add experiences to buffer
                for exp in batch:
                    try:
                        self.buffer.put(exp, timeout=0.1)
                    except queue.Full:
                        # Buffer full, drop experience
                        print("Experience buffer full, dropping experience")
                
        except Exception as e:
            print(f"Error in streaming thread: {e}")
        finally:
            print("Experience streaming stopped")
    
    def close(self):
        """Close the consumer and stop streaming."""
        self.stop_streaming()
        self.consumer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Utility functions for common use cases

def create_replay_buffer(capacity: int = 100000) -> 'PrioritizedReplayBuffer':
    """Create a prioritized replay buffer for experience replay.
    
    Args:
        capacity: Maximum number of experiences to store
        
    Returns:
        PrioritizedReplayBuffer instance
    """
    # This would be implemented with a proper replay buffer
    # For now, just a placeholder
    raise NotImplementedError("Replay buffer implementation needed")


def experience_to_tensors(
    experiences: List[Experience]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a batch of experiences to training tensors.
    
    Args:
        experiences: List of Experience objects
        
    Returns:
        Tuple of (states, actions, rewards, next_states, dones, action_masks)
    """
    if not experiences:
        raise ValueError("Empty experience list")
    
    states = np.stack([exp.state for exp in experiences])
    actions = np.array([exp.action for exp in experiences], dtype=np.int32)
    rewards = np.array([exp.reward for exp in experiences], dtype=np.float32)
    next_states = np.stack([exp.next_state for exp in experiences])
    dones = np.array([exp.done for exp in experiences], dtype=np.bool_)
    action_masks = np.stack([exp.action_mask for exp in experiences])
    
    return states, actions, rewards, next_states, dones, action_masks