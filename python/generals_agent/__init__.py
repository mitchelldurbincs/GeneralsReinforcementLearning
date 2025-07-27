"""Generals.io Agent Framework"""

# Core components
from .base_agent import BaseAgent
from .random_agent import RandomAgent

# New architecture components
from .connection import GameConnection
from .game_client import GameClient, PlayerCredentials, GameConfig
from .errors import (
    GameError, ConnectionError, GameStateError, ActionError,
    AuthenticationError, GameNotFoundError, GameFullError,
    TurnTimeoutError, InvalidMoveError
)
from .types import (
    AgentState, GameStatus, AgentContext, Position, Move,
    GameStats, PlayerStats, TileInfo, BoardView
)
from .game_session import GameSession
from .events import GameEventDispatcher, BaseGameEventHandler, LoggingEventHandler
from .polling import (
    PollingStrategy, FixedIntervalPolling, ExponentialBackoffPolling,
    AdaptivePolling, TurnBasedPolling, CompositePolling
)

__all__ = [
    # Original exports
    'BaseAgent', 'RandomAgent',
    
    # New architecture exports
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