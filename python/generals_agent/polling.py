"""
Configurable polling strategies for the Generals.io agent framework.
Provides different approaches for polling game state updates.
"""
from abc import ABC, abstractmethod
from typing import Optional
import time
import random


class PollingStrategy(ABC):
    """Abstract base class for polling strategies."""
    
    @abstractmethod
    def get_next_delay(self, attempt: int, last_error: Optional[Exception] = None) -> float:
        """
        Get the delay before the next poll attempt.
        
        Args:
            attempt: The current attempt number (starts at 0)
            last_error: The last error that occurred, if any
            
        Returns:
            The delay in seconds before the next poll
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the polling strategy to its initial state."""
        pass


class FixedIntervalPolling(PollingStrategy):
    """Poll at a fixed interval regardless of conditions."""
    
    def __init__(self, interval: float = 0.1):
        """
        Initialize with a fixed polling interval.
        
        Args:
            interval: The fixed delay between polls in seconds
        """
        self.interval = interval
    
    def get_next_delay(self, attempt: int, last_error: Optional[Exception] = None) -> float:
        """Return the fixed interval."""
        return self.interval
    
    def reset(self) -> None:
        """No state to reset for fixed interval polling."""
        pass


class ExponentialBackoffPolling(PollingStrategy):
    """
    Poll with exponential backoff on errors.
    Useful for handling temporary failures gracefully.
    """
    
    def __init__(self, base_interval: float = 0.1, max_interval: float = 5.0, 
                 multiplier: float = 2.0, jitter: bool = True):
        """
        Initialize exponential backoff polling.
        
        Args:
            base_interval: The base interval in seconds
            max_interval: Maximum interval in seconds
            multiplier: The backoff multiplier
            jitter: Whether to add random jitter to prevent thundering herd
        """
        self.base_interval = base_interval
        self.max_interval = max_interval
        self.multiplier = multiplier
        self.jitter = jitter
        self._consecutive_errors = 0
    
    def get_next_delay(self, attempt: int, last_error: Optional[Exception] = None) -> float:
        """Calculate delay with exponential backoff on errors."""
        if last_error is not None:
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0
        
        if self._consecutive_errors == 0:
            delay = self.base_interval
        else:
            delay = min(
                self.base_interval * (self.multiplier ** self._consecutive_errors),
                self.max_interval
            )
        
        if self.jitter and delay > self.base_interval:
            # Add random jitter up to 25% of the delay
            jitter_amount = delay * 0.25 * random.random()
            delay = delay - (jitter_amount / 2) + (jitter_amount * random.random())
        
        return delay
    
    def reset(self) -> None:
        """Reset error count."""
        self._consecutive_errors = 0


class AdaptivePolling(PollingStrategy):
    """
    Adaptive polling that adjusts based on game activity.
    Polls more frequently during active turns, less during waiting.
    """
    
    def __init__(self, active_interval: float = 0.05, 
                 waiting_interval: float = 1.0,
                 transition_steps: int = 5):
        """
        Initialize adaptive polling.
        
        Args:
            active_interval: Interval during active gameplay
            waiting_interval: Interval while waiting for players or turns
            transition_steps: Number of polls to transition between intervals
        """
        self.active_interval = active_interval
        self.waiting_interval = waiting_interval
        self.transition_steps = transition_steps
        
        self._current_interval = waiting_interval
        self._target_interval = waiting_interval
        self._is_active = False
        self._polls_since_change = 0
    
    def set_game_active(self, is_active: bool) -> None:
        """
        Signal whether the game is in an active state.
        
        Args:
            is_active: True if game is active, False if waiting
        """
        self._is_active = is_active
        self._target_interval = self.active_interval if is_active else self.waiting_interval
        self._polls_since_change = 0
    
    def get_next_delay(self, attempt: int, last_error: Optional[Exception] = None) -> float:
        """Get delay that adapts to game state."""
        # Gradually transition to target interval
        if self._current_interval != self._target_interval:
            self._polls_since_change += 1
            
            if self._polls_since_change <= self.transition_steps:
                # Linear interpolation
                progress = self._polls_since_change / self.transition_steps
                self._current_interval = (
                    self._current_interval * (1 - progress) + 
                    self._target_interval * progress
                )
            else:
                self._current_interval = self._target_interval
        
        return self._current_interval
    
    def reset(self) -> None:
        """Reset to initial waiting state."""
        self._current_interval = self.waiting_interval
        self._target_interval = self.waiting_interval
        self._is_active = False
        self._polls_since_change = 0


class TurnBasedPolling(PollingStrategy):
    """
    Polling strategy optimized for turn-based games.
    Polls frequently near turn deadlines, less frequently otherwise.
    """
    
    def __init__(self, turn_duration: float = 0.5,
                 early_interval: float = 0.2,
                 urgent_interval: float = 0.05,
                 urgent_threshold: float = 0.1):
        """
        Initialize turn-based polling.
        
        Args:
            turn_duration: Expected duration of a turn in seconds
            early_interval: Polling interval early in the turn
            urgent_interval: Polling interval near turn deadline
            urgent_threshold: Time before deadline to start urgent polling
        """
        self.turn_duration = turn_duration
        self.early_interval = early_interval
        self.urgent_interval = urgent_interval
        self.urgent_threshold = urgent_threshold
        
        self._turn_start_time: Optional[float] = None
    
    def signal_turn_start(self) -> None:
        """Signal that a new turn has started."""
        self._turn_start_time = time.time()
    
    def get_next_delay(self, attempt: int, last_error: Optional[Exception] = None) -> float:
        """Get delay based on time remaining in turn."""
        if self._turn_start_time is None:
            return self.early_interval
        
        elapsed = time.time() - self._turn_start_time
        remaining = self.turn_duration - elapsed
        
        if remaining <= self.urgent_threshold:
            return self.urgent_interval
        else:
            return self.early_interval
    
    def reset(self) -> None:
        """Reset turn timing."""
        self._turn_start_time = None


class CompositePolling(PollingStrategy):
    """
    Composite polling strategy that combines multiple strategies.
    Useful for complex polling requirements.
    """
    
    def __init__(self, strategies: list[tuple[PollingStrategy, float]]):
        """
        Initialize with weighted strategies.
        
        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies
        self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total_weight = sum(weight for _, weight in self.strategies)
        if total_weight > 0:
            self.strategies = [
                (strategy, weight / total_weight)
                for strategy, weight in self.strategies
            ]
    
    def get_next_delay(self, attempt: int, last_error: Optional[Exception] = None) -> float:
        """Get weighted average delay from all strategies."""
        total_delay = 0.0
        
        for strategy, weight in self.strategies:
            delay = strategy.get_next_delay(attempt, last_error)
            total_delay += delay * weight
        
        return total_delay
    
    def reset(self) -> None:
        """Reset all component strategies."""
        for strategy, _ in self.strategies:
            strategy.reset()