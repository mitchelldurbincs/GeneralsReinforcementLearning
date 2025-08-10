"""Generals.io Agent Framework"""

# Core components
from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .agent_runner import AgentRunner

# Infrastructure components
from .connection import GameConnection
from .game_client import GameClient, PlayerCredentials, GameConfig
from .game_session import GameSession

# Event system
from .events import GameEventDispatcher, BaseGameEventHandler, LoggingEventHandler

# Polling strategies
from .polling import (
    PollingStrategy, FixedIntervalPolling, ExponentialBackoffPolling,
    AdaptivePolling, TurnBasedPolling, CompositePolling
)

# Error types
from .errors import (
    GameError, ConnectionError, GameStateError, ActionError,
    AuthenticationError, GameNotFoundError, GameFullError,
    TurnTimeoutError, InvalidMoveError
)

# Type definitions
from .types import (
    AgentState, GameStatus, AgentContext, Position, Move,
    GameStats, PlayerStats, TileInfo, BoardView
)

__all__ = [
    # Core exports
    'BaseAgent', 'RandomAgent', 'AgentRunner',
    
    # Infrastructure
    'GameConnection', 'GameClient', 'PlayerCredentials', 'GameConfig',
    'GameSession',
    
    # Event system
    'GameEventDispatcher', 'BaseGameEventHandler', 'LoggingEventHandler',
    
    # Polling strategies
    'PollingStrategy', 'FixedIntervalPolling', 'ExponentialBackoffPolling',
    'AdaptivePolling', 'TurnBasedPolling', 'CompositePolling',
    
    # Error types
    'GameError', 'ConnectionError', 'GameStateError', 'ActionError',
    'AuthenticationError', 'GameNotFoundError', 'GameFullError',
    'TurnTimeoutError', 'InvalidMoveError',
    
    # Type definitions
    'AgentState', 'GameStatus', 'AgentContext', 'Position', 'Move',
    'GameStats', 'PlayerStats', 'TileInfo', 'BoardView'
]