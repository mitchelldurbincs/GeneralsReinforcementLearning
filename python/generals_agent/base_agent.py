"""
New simplified BaseAgent interface focused only on agent behavior.
This will eventually replace the current base_agent.py.
"""
from abc import ABC, abstractmethod
from typing import Optional
import logging

from .types import BoardView, Move, Position

# Import protobuf types
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))
from generals_pb.game.v1 import game_pb2


class BaseAgent(ABC):
    """
    Pure abstract agent interface focused only on game behavior.
    
    This simplified interface removes all networking, state management,
    and lifecycle concerns, leaving only the core agent decision-making logic.
    """
    
    def __init__(self, name: str = "BaseAgent"):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent for logging and identification
        """
        self.name = name
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{name}]")
        self.player_id: Optional[str] = None
    
    def set_player_id(self, player_id: str):
        """
        Set the player ID for this agent.
        Called by the runner when the agent joins a game.
        
        Args:
            player_id: The assigned player ID
        """
        self.player_id = player_id
        self.logger.info(f"Assigned player ID: {player_id}")
    
    @abstractmethod
    def select_action(self, game_state: game_pb2.GameState) -> Optional[game_pb2.Action]:
        """
        Select an action based on the current game state.
        
        This is the core method that agents must implement. It should analyze
        the game state and return an appropriate action, or None to pass.
        
        Args:
            game_state: The current game state from the agent's perspective
            
        Returns:
            An Action to perform, or None to pass this turn
        """
        pass
    
    def select_action_from_board_view(self, board_view: BoardView) -> Optional[Move]:
        """
        Convenience method to select action using simplified BoardView.
        
        Subclasses can override this instead of select_action for easier
        implementation. By default, this is not implemented.
        
        Args:
            board_view: Simplified view of the game board
            
        Returns:
            A Move to perform, or None to pass this turn
        """
        return None
    
    @abstractmethod
    def on_game_start(self, initial_state: game_pb2.GameState):
        """
        Called when the game starts.
        
        Agents can use this to perform any initialization based on the
        initial game state (e.g., analyzing the map, finding their general).
        
        Args:
            initial_state: The initial game state
        """
        pass
    
    @abstractmethod
    def on_game_end(self, final_state: game_pb2.GameState, winner_id: str):
        """
        Called when the game ends.
        
        Agents can use this for learning, logging statistics, or cleanup.
        
        Args:
            final_state: The final game state
            winner_id: ID of the winning player
        """
        pass
    
    def on_turn_start(self, turn: int):
        """
        Called at the start of each turn.
        
        Optional method that agents can override for turn-based logic.
        
        Args:
            turn: The current turn number
        """
        pass
    
    def on_turn_end(self, turn: int):
        """
        Called at the end of each turn.
        
        Optional method that agents can override for turn-based cleanup.
        
        Args:
            turn: The current turn number
        """
        pass
    
    def on_error(self, error: Exception):
        """
        Called when an error occurs.
        
        Optional method that agents can override for error handling.
        
        Args:
            error: The exception that occurred
        """
        self.logger.error(f"Error occurred: {error}")
    
    def get_name(self) -> str:
        """Get the agent's name."""
        return self.name
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name={self.name})"