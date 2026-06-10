"""Thread-safe experience replay buffer for parallel collection.

Promoted from the single-threaded ReplayBuffer in train_dqn_agent.py so that
multiple environment worker threads can push transitions while a learner
thread samples batches.
"""

import random
import threading
from typing import List, Tuple


class ReplayBuffer:
    """Thread-safe ring-buffer replay memory.

    Backed by a list rather than a deque: random.sample on a deque costs
    O(n) per draw while the lock is held, on a list it is O(1).
    """

    def __init__(self, capacity: int):
        """Initialize replay buffer with given capacity."""
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._buffer: List[Tuple] = []
        self._write_idx = 0
        self._total_pushed = 0
        self._lock = threading.Lock()

    def push(self, state, action, reward, next_state, done) -> None:
        """Add an experience to the buffer, evicting the oldest when full."""
        item = (state, action, reward, next_state, done)
        with self._lock:
            if len(self._buffer) < self.capacity:
                self._buffer.append(item)
            else:
                self._buffer[self._write_idx] = item
            self._write_idx = (self._write_idx + 1) % self.capacity
            self._total_pushed += 1

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences uniformly at random."""
        with self._lock:
            return random.sample(self._buffer, batch_size)

    @property
    def total_pushed(self) -> int:
        """Monotonic count of pushes; doubles as a global env-step counter
        (one transition pushed == one environment step)."""
        with self._lock:
            return self._total_pushed

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
