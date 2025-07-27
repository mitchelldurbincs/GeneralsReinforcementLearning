"""
Event-driven architecture for the Generals.io agent framework.
Provides event dispatching and handler registration.
"""
from typing import Protocol, List, Optional, Callable, Any
from abc import abstractmethod
import logging

from .types import AgentState, GameStatus
from .errors import GameError

# Import protobuf types
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))
from generals_pb.game.v1 import game_pb2


class GameEventHandler(Protocol):
    """Protocol defining the interface for game event handlers."""
    
    @abstractmethod
    def on_state_update(self, old_state: Optional[game_pb2.GameState], 
                       new_state: game_pb2.GameState) -> None:
        """Called when game state is updated."""
        ...
    
    @abstractmethod
    def on_turn_change(self, turn: int) -> None:
        """Called when turn number changes."""
        ...
    
    @abstractmethod
    def on_game_start(self, initial_state: game_pb2.GameState) -> None:
        """Called when game transitions from waiting to in progress."""
        ...
    
    @abstractmethod
    def on_game_end(self, final_state: game_pb2.GameState, winner_id: str) -> None:
        """Called when game ends."""
        ...
    
    @abstractmethod
    def on_player_joined(self, player_id: str, player_name: str) -> None:
        """Called when a new player joins the game."""
        ...
    
    @abstractmethod
    def on_player_left(self, player_id: str) -> None:
        """Called when a player leaves the game."""
        ...
    
    @abstractmethod
    def on_error(self, error: Exception) -> None:
        """Called when an error occurs."""
        ...


class BaseGameEventHandler:
    """
    Base implementation of GameEventHandler with no-op methods.
    Subclasses can override only the events they care about.
    """
    
    def on_state_update(self, old_state: Optional[game_pb2.GameState], 
                       new_state: game_pb2.GameState) -> None:
        """Default no-op implementation."""
        pass
    
    def on_turn_change(self, turn: int) -> None:
        """Default no-op implementation."""
        pass
    
    def on_game_start(self, initial_state: game_pb2.GameState) -> None:
        """Default no-op implementation."""
        pass
    
    def on_game_end(self, final_state: game_pb2.GameState, winner_id: str) -> None:
        """Default no-op implementation."""
        pass
    
    def on_player_joined(self, player_id: str, player_name: str) -> None:
        """Default no-op implementation."""
        pass
    
    def on_player_left(self, player_id: str) -> None:
        """Default no-op implementation."""
        pass
    
    def on_error(self, error: Exception) -> None:
        """Default no-op implementation."""
        pass


class GameEventDispatcher:
    """
    Central event dispatcher for game events.
    Manages event handler registration and event distribution.
    """
    
    def __init__(self):
        """Initialize the event dispatcher."""
        self.handlers: List[GameEventHandler] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self._error_callback: Optional[Callable[[GameEventHandler, Exception], None]] = None
    
    def register(self, handler: GameEventHandler) -> None:
        """
        Register a new event handler.
        
        Args:
            handler: The event handler to register
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
            self.logger.debug(f"Registered handler: {handler.__class__.__name__}")
    
    def unregister(self, handler: GameEventHandler) -> None:
        """
        Unregister an event handler.
        
        Args:
            handler: The event handler to unregister
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
            self.logger.debug(f"Unregistered handler: {handler.__class__.__name__}")
    
    def set_error_callback(self, callback: Callable[[GameEventHandler, Exception], None]) -> None:
        """
        Set a callback for handling errors from event handlers.
        
        Args:
            callback: Function to call when a handler raises an exception
        """
        self._error_callback = callback
    
    def dispatch_state_update(self, old_state: Optional[game_pb2.GameState], 
                            new_state: game_pb2.GameState) -> None:
        """Dispatch state update event to all handlers."""
        self._dispatch_event('on_state_update', old_state, new_state)
    
    def dispatch_turn_change(self, turn: int) -> None:
        """Dispatch turn change event to all handlers."""
        self._dispatch_event('on_turn_change', turn)
    
    def dispatch_game_start(self, initial_state: game_pb2.GameState) -> None:
        """Dispatch game start event to all handlers."""
        self._dispatch_event('on_game_start', initial_state)
    
    def dispatch_game_end(self, final_state: game_pb2.GameState, winner_id: str) -> None:
        """Dispatch game end event to all handlers."""
        self._dispatch_event('on_game_end', final_state, winner_id)
    
    def dispatch_player_joined(self, player_id: str, player_name: str) -> None:
        """Dispatch player joined event to all handlers."""
        self._dispatch_event('on_player_joined', player_id, player_name)
    
    def dispatch_player_left(self, player_id: str) -> None:
        """Dispatch player left event to all handlers."""
        self._dispatch_event('on_player_left', player_id)
    
    def dispatch_error(self, error: Exception) -> None:
        """Dispatch error event to all handlers."""
        self._dispatch_event('on_error', error)
    
    def _dispatch_event(self, event_name: str, *args, **kwargs) -> None:
        """
        Internal method to dispatch an event to all handlers.
        
        Args:
            event_name: Name of the event method to call
            *args: Arguments to pass to the event method
            **kwargs: Keyword arguments to pass to the event method
        """
        for handler in self.handlers:
            try:
                method = getattr(handler, event_name)
                method(*args, **kwargs)
            except Exception as e:
                self._handle_handler_error(handler, e, event_name)
    
    def _handle_handler_error(self, handler: GameEventHandler, error: Exception, 
                            event_name: str) -> None:
        """Handle errors from event handlers."""
        self.logger.error(
            f"Error in handler {handler.__class__.__name__}.{event_name}: {error}",
            exc_info=True
        )
        
        if self._error_callback:
            try:
                self._error_callback(handler, error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}", exc_info=True)


class LoggingEventHandler(BaseGameEventHandler):
    """Event handler that logs all events for debugging."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with optional logger."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def on_state_update(self, old_state: Optional[game_pb2.GameState], 
                       new_state: game_pb2.GameState) -> None:
        """Log state updates."""
        self.logger.debug(f"State update: turn {new_state.turn}")
    
    def on_turn_change(self, turn: int) -> None:
        """Log turn changes."""
        self.logger.info(f"Turn changed to {turn}")
    
    def on_game_start(self, initial_state: game_pb2.GameState) -> None:
        """Log game start."""
        self.logger.info("Game started!")
    
    def on_game_end(self, final_state: game_pb2.GameState, winner_id: str) -> None:
        """Log game end."""
        self.logger.info(f"Game ended. Winner: {winner_id}")
    
    def on_player_joined(self, player_id: str, player_name: str) -> None:
        """Log player joins."""
        self.logger.info(f"Player joined: {player_name} ({player_id})")
    
    def on_player_left(self, player_id: str) -> None:
        """Log player leaves."""
        self.logger.info(f"Player left: {player_id}")
    
    def on_error(self, error: Exception) -> None:
        """Log errors."""
        self.logger.error(f"Error occurred: {error}", exc_info=True)