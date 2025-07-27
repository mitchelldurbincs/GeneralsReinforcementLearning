"""Generals.io Agent Framework"""

# Core components - Original
from .base_agent import BaseAgent
from .random_agent import RandomAgent

# Core components - New architecture
from .base_agent_new import BaseAgentNew
from .agent_runner import AgentRunner
from .base_agent_adapter import BaseAgentAdapter, run_old_agent_with_new_architecture
from .random_agent_new import RandomAgentNew

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
    # Original exports
    'BaseAgent', 'RandomAgent',
    
    # New architecture core
    'BaseAgentNew', 'AgentRunner', 'BaseAgentAdapter', 
    'run_old_agent_with_new_architecture', 'RandomAgentNew',
    
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