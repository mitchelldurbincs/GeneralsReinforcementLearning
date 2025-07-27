"""
GameSession class for managing a single game session lifecycle.
Handles polling, state updates, and player coordination.
"""
import time
import logging
from typing import Callable, Optional, List
from dataclasses import dataclass

from .game_client import GameClient, PlayerCredentials
from .types import AgentContext, AgentState, GameStatus, BoardView
from .errors import GameError, GameStateError, TurnTimeoutError
from .polling import PollingStrategy, FixedIntervalPolling
from .events import GameEventDispatcher

# Import protobuf types
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))
from generals_pb.game.v1 import game_pb2


class GameSession:
    """
    Manages a single game session lifecycle.
    
    Provides polling mechanisms, state tracking, and event callbacks
    for a single game from join to completion.
    """
    
    def __init__(self, client: GameClient, credentials: PlayerCredentials, 
                 polling_strategy: Optional[PollingStrategy] = None,
                 event_dispatcher: Optional[GameEventDispatcher] = None):
        """
        Initialize a game session.
        
        Args:
            client: The GameClient instance for RPC calls
            credentials: Player credentials for this session
            polling_strategy: Strategy for polling intervals (defaults to FixedIntervalPolling)
            event_dispatcher: Event dispatcher for game events
        """
        self.client = client
        self.credentials = credentials
        self.polling_strategy = polling_strategy or FixedIntervalPolling(0.1)
        self.event_dispatcher = event_dispatcher or GameEventDispatcher()
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{credentials.game_id[:8]}]")
        
        # Session state
        self.context = AgentContext(
            state=AgentState.WAITING_FOR_GAME,
            game_id=credentials.game_id,
            player_id=credentials.player_id,
            player_token=credentials.player_token
        )
        
        # Callbacks
        self._state_update_callback: Optional[Callable[[game_pb2.GameState], None]] = None
        self._turn_change_callback: Optional[Callable[[int], None]] = None
        self._game_start_callback: Optional[Callable[[game_pb2.GameState], None]] = None
        self._game_end_callback: Optional[Callable[[game_pb2.GameState, str], None]] = None
        
        # Control flags
        self._polling = False
        self._last_known_state: Optional[game_pb2.GameState] = None
    
    def poll_updates(self, callback: Callable[[game_pb2.GameState], None]):
        """
        Start polling for game state updates.
        
        Args:
            callback: Function to call with each new game state
        """
        self._state_update_callback = callback
        self._polling = True
        
        self.logger.info("Starting game session polling")
        
        attempt = 0
        last_error = None
        
        while self._polling and self.is_active():
            try:
                # Get current game state
                game_state = self.client.get_game_state(self.credentials)
                
                # Update context
                self.context.current_game_state = game_state
                
                # Check for state changes
                self._process_state_update(game_state)
                
                # Reset error state on success
                last_error = None
                attempt = 0
                
                # Get delay from polling strategy
                delay = self.polling_strategy.get_next_delay(attempt, last_error)
                time.sleep(delay)
                
            except TurnTimeoutError as e:
                self.logger.warning("Turn timeout occurred")
                last_error = e
                attempt += 1
                self.event_dispatcher.dispatch_error(e)
                # Continue polling, game might still be active
                
            except GameStateError as e:
                self.logger.error(f"Game state error: {e}")
                self.context.state = AgentState.ERROR
                self.context.error = e
                self.event_dispatcher.dispatch_error(e)
                break
                
            except Exception as e:
                self.logger.error(f"Unexpected error during polling: {e}")
                self.context.state = AgentState.ERROR
                self.context.error = e
                last_error = e
                attempt += 1
                self.event_dispatcher.dispatch_error(e)
                
                # Give up after too many attempts
                if attempt > 10:
                    break
    
    def stop_polling(self):
        """Stop the polling loop."""
        self._polling = False
        self.logger.info("Stopped polling")
    
    def wait_for_players(self, min_players: int = 2, timeout: float = 300.0):
        """
        Wait for minimum number of players to join the game.
        
        Args:
            min_players: Minimum number of players required
            timeout: Maximum time to wait in seconds
            
        Raises:
            TimeoutError: If timeout is reached before enough players join
        """
        self.logger.info(f"Waiting for at least {min_players} players")
        self.context.state = AgentState.WAITING_FOR_PLAYERS
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                game_state = self.client.get_game_state(self.credentials)
                
                # Count active players
                active_players = sum(1 for p in game_state.players if p.is_active)
                
                if active_players >= min_players:
                    self.logger.info(f"Found {active_players} players, starting game")
                    return
                
                # Check if game already started
                if game_state.status == game_pb2.GameStatus.IN_PROGRESS:
                    self.logger.info("Game already in progress")
                    return
                
                time.sleep(1.0)  # Check less frequently while waiting
                
            except Exception as e:
                self.logger.error(f"Error while waiting for players: {e}")
                raise
        
        raise TimeoutError(f"Timeout waiting for {min_players} players after {timeout}s")
    
    def is_active(self) -> bool:
        """Check if the game session is still active."""
        return self.context.state in [
            AgentState.WAITING_FOR_GAME,
            AgentState.WAITING_FOR_PLAYERS, 
            AgentState.IN_GAME
        ]
    
    def get_board_view(self) -> Optional[BoardView]:
        """
        Get a simplified view of the current board state.
        
        Returns:
            BoardView if game state is available, None otherwise
        """
        if self.context.current_game_state:
            return BoardView.from_game_state(
                self.context.current_game_state,
                self.context.player_id
            )
        return None
    
    def _process_state_update(self, game_state: game_pb2.GameState):
        """Process a game state update and trigger appropriate callbacks."""
        # Store previous state for comparison
        old_state = self._last_known_state
        
        # Detect game status changes
        game_status = self._get_game_status(game_state)
        
        if self.context.last_game_status != game_status:
            self._handle_status_change(self.context.last_game_status, game_status, game_state)
            self.context.last_game_status = game_status
        
        # Detect turn changes
        if game_state.turn != self.context.last_turn:
            old_turn = self.context.last_turn
            self.context.last_turn = game_state.turn
            
            # Dispatch turn change events
            self.event_dispatcher.dispatch_turn_change(game_state.turn)
            if self._turn_change_callback:
                self._turn_change_callback(game_state.turn)
        
        # Detect player changes
        if old_state:
            self._detect_player_changes(old_state, game_state)
        
        # Dispatch state update event
        self.event_dispatcher.dispatch_state_update(old_state, game_state)
        
        # Always call state update callback
        if self._state_update_callback:
            self._state_update_callback(game_state)
        
        self._last_known_state = game_state
    
    def _get_game_status(self, game_state: game_pb2.GameState) -> GameStatus:
        """Convert protobuf game status to our enum."""
        if game_state.status == game_pb2.GameStatus.WAITING:
            return GameStatus.WAITING
        elif game_state.status == game_pb2.GameStatus.IN_PROGRESS:
            return GameStatus.IN_PROGRESS
        elif game_state.status == game_pb2.GameStatus.FINISHED:
            return GameStatus.FINISHED
        else:
            return GameStatus.CANCELLED
    
    def _handle_status_change(self, old_status: Optional[GameStatus], 
                            new_status: GameStatus, 
                            game_state: game_pb2.GameState):
        """Handle game status transitions."""
        self.logger.info(f"Game status changed: {old_status} -> {new_status}")
        
        # Game started
        if old_status in [None, GameStatus.WAITING] and new_status == GameStatus.IN_PROGRESS:
            self.context.state = AgentState.IN_GAME
            self.event_dispatcher.dispatch_game_start(game_state)
            if self._game_start_callback:
                self._game_start_callback(game_state)
        
        # Game ended
        elif new_status == GameStatus.FINISHED:
            self.context.state = AgentState.GAME_ENDED
            winner_id = self._find_winner(game_state)
            self.event_dispatcher.dispatch_game_end(game_state, winner_id)
            if self._game_end_callback:
                self._game_end_callback(game_state, winner_id)
            self.stop_polling()
    
    def _find_winner(self, game_state: game_pb2.GameState) -> str:
        """Find the winner from the final game state."""
        # Look for the last active player
        active_players = [p for p in game_state.players if p.is_active]
        if len(active_players) == 1:
            return active_players[0].id
        
        # If multiple players active, find one with most tiles
        if active_players:
            return max(active_players, key=lambda p: p.tile_count).id
        
        return ""  # No winner
    
    def _detect_player_changes(self, old_state: game_pb2.GameState, 
                              new_state: game_pb2.GameState):
        """Detect players joining or leaving the game."""
        old_players = {p.id: p for p in old_state.players}
        new_players = {p.id: p for p in new_state.players}
        
        # Check for new players
        for player_id, player in new_players.items():
            if player_id not in old_players:
                self.event_dispatcher.dispatch_player_joined(player_id, player.name)
        
        # Check for players who left
        for player_id in old_players:
            if player_id not in new_players:
                self.event_dispatcher.dispatch_player_left(player_id)
    
    # Event handler setters
    def on_turn_change(self, callback: Callable[[int], None]):
        """Set callback for turn changes."""
        self._turn_change_callback = callback
    
    def on_game_start(self, callback: Callable[[game_pb2.GameState], None]):
        """Set callback for game start."""
        self._game_start_callback = callback
    
    def on_game_end(self, callback: Callable[[game_pb2.GameState, str], None]):
        """Set callback for game end."""
        self._game_end_callback = callback