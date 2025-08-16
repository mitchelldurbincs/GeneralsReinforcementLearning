"""
AgentRunner class that orchestrates the agent lifecycle.
Handles all the networking, polling, and state management so agents can focus on game logic.
"""
import logging
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from .base_agent import BaseAgent
from .connection import GameConnection
from .game_client import GameClient, PlayerCredentials, GameConfig
from .game_session import GameSession
from .events import BaseGameEventHandler, GameEventDispatcher, LoggingEventHandler
from .polling import PollingStrategy, FixedIntervalPolling, AdaptivePolling
from .types import AgentState, BoardView, Move
from .errors import GameError, ActionError

# Import protobuf types
from generals_pb.game.v1 import game_pb2


class AgentEventHandler(BaseGameEventHandler):
    """Event handler that bridges game events to agent methods."""
    
    def __init__(self, agent: BaseAgent, runner: 'AgentRunner'):
        """Initialize the handler."""
        self.agent = agent
        self.runner = runner
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{agent.name}]")
    
    def on_state_update(self, old_state: Optional[game_pb2.GameState], 
                       new_state: game_pb2.GameState) -> None:
        """Handle state updates by asking agent for actions."""
        # Only process if it's our turn
        if not self._is_our_turn(new_state):
            return
        
        try:
            # Notify agent of turn start
            self.agent.on_turn_start(new_state.turn)
            
            # Get action from agent
            action = self.agent.select_action(new_state)
            
            # If agent didn't implement select_action but did implement select_action_from_board_view
            if action is None and hasattr(self.agent, 'select_action_from_board_view'):
                board_view = BoardView.from_game_state(new_state, self.agent.player_id)
                move = self.agent.select_action_from_board_view(board_view)
                if move:
                    action = move.to_proto()
            
            # Submit action if we got one
            if action:
                self.runner.submit_action(action)
            else:
                self.logger.debug(f"Agent passed on turn {new_state.turn}")
            
            # Notify agent of turn end
            self.agent.on_turn_end(new_state.turn)
            
        except Exception as e:
            self.logger.error(f"Error in agent action selection: {e}", exc_info=True)
            self.agent.on_error(e)
    
    def on_game_start(self, initial_state: game_pb2.GameState) -> None:
        """Notify agent of game start."""
        self.agent.on_game_start(initial_state)
    
    def on_game_end(self, final_state: game_pb2.GameState, winner_id: str) -> None:
        """Notify agent of game end."""
        self.agent.on_game_end(final_state, winner_id)
    
    def on_error(self, error: Exception) -> None:
        """Notify agent of errors."""
        self.agent.on_error(error)
    
    def _is_our_turn(self, game_state: game_pb2.GameState) -> bool:
        """Check if it's our turn to move."""
        # In Generals.io, all players move simultaneously each turn
        # So we should submit an action every turn while the game is active
        from generals_pb.common.v1 import common_pb2
        return (game_state.status == common_pb2.GAME_STATUS_IN_PROGRESS and
                self.agent.player_id is not None)


class AgentRunner:
    """
    Orchestrates the lifecycle of an agent.
    
    Handles connection management, game joining, polling, and action submission
    so that agents can focus purely on game logic.
    """
    
    def __init__(self, agent: BaseAgent, server_address: str = "localhost:50051",
                 polling_strategy: Optional[PollingStrategy] = None,
                 enable_logging: bool = True):
        """
        Initialize the agent runner.
        
        Args:
            agent: The agent to run
            server_address: gRPC server address
            polling_strategy: Strategy for polling (defaults to AdaptivePolling)
            enable_logging: Whether to enable debug event logging
        """
        self.agent = agent
        self.server_address = server_address
        self.polling_strategy = polling_strategy or AdaptivePolling()
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{agent.name}]")
        
        # Components
        self.connection: Optional[GameConnection] = None
        self.client: Optional[GameClient] = None
        self.session: Optional[GameSession] = None
        self.credentials: Optional[PlayerCredentials] = None
        
        # Event handling
        self.event_dispatcher = GameEventDispatcher()
        self.event_handler = AgentEventHandler(agent, self)
        self.event_dispatcher.register(self.event_handler)
        
        if enable_logging:
            self.event_dispatcher.register(LoggingEventHandler())
        
        # State
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._polling_future: Optional[Future] = None
    
    def connect(self) -> GameClient:
        """Establish connection to the game server."""
        if self.connection is None:
            self.logger.info(f"Connecting to {self.server_address}")
            self.connection = GameConnection(self.server_address)
            self.client = GameClient(self.connection)
            self.connection.connect()
        return self.client
    
    def disconnect(self):
        """Disconnect from the game server."""
        if self.connection:
            self.connection.disconnect()
            self.connection = None
            self.client = None
    
    def create_game(self, config: GameConfig) -> str:
        """
        Create a new game.
        
        Args:
            config: Game configuration
            
        Returns:
            The created game ID
        """
        client = self.connect()
        game_id = client.create_game(config)
        self.logger.info(f"Created game: {game_id}")
        return game_id
    
    def join_game(self, game_id: str) -> PlayerCredentials:
        """
        Join an existing game.
        
        Args:
            game_id: The game to join
            
        Returns:
            Player credentials
        """
        client = self.connect()
        self.credentials = client.join_game(game_id, self.agent.name)
        self.agent.set_player_id(self.credentials.player_id)
        self.logger.info(f"Joined game {game_id} as player {self.credentials.player_id}")
        return self.credentials
    
    def create_and_join_game(self, config: GameConfig) -> PlayerCredentials:
        """
        Create a new game and join it.
        
        Args:
            config: Game configuration
            
        Returns:
            Player credentials
        """
        game_id = self.create_game(config)
        return self.join_game(game_id)
    
    def wait_for_players(self, min_players: int = 2, timeout: float = 300.0):
        """
        Wait for other players to join.
        
        Args:
            min_players: Minimum number of players required
            timeout: Maximum time to wait in seconds
        """
        if not self.session:
            raise RuntimeError("Must join a game before waiting for players")
        
        self.session.wait_for_players(min_players, timeout)
    
    def run(self, wait_for_players: int = 2, timeout: float = 300.0):
        """
        Run the agent for a single game.
        
        This is the main entry point that handles the complete game lifecycle.
        
        Args:
            wait_for_players: Number of players to wait for before starting
            timeout: Timeout for waiting for players
        """
        if not self.credentials:
            raise RuntimeError("Must join a game before running")
        
        try:
            self._running = True
            
            # Create game session
            self.session = GameSession(
                self.client,
                self.credentials,
                self.polling_strategy,
                self.event_dispatcher
            )
            
            # Wait for players if needed
            if wait_for_players > 1:
                self.logger.info(f"Waiting for {wait_for_players} players...")
                self.session.wait_for_players(wait_for_players, timeout)
            
            # Start polling in background thread
            self._polling_future = self._executor.submit(self._run_polling)
            
            # Wait for game to end
            while self._running and self.session.is_active():
                time.sleep(0.1)
            
            self.logger.info("Game ended")
            
        except Exception as e:
            self.logger.error(f"Error during agent run: {e}", exc_info=True)
            raise
        finally:
            self.stop()
    
    def run_async(self, wait_for_players: int = 2, timeout: float = 300.0) -> Future:
        """
        Run the agent asynchronously.
        
        Returns:
            A Future that completes when the game ends
        """
        return self._executor.submit(self.run, wait_for_players, timeout)
    
    def stop(self):
        """Stop the agent runner."""
        self._running = False
        
        if self.session:
            self.session.stop_polling()
        
        if self._polling_future:
            self._polling_future.result(timeout=5.0)
        
        # Note: leave_game is not implemented in the current protobuf
        # if self.credentials and self.client:
        #     try:
        #         self.client.leave_game(self.credentials)
        #     except Exception as e:
        #         self.logger.error(f"Error leaving game: {e}")
    
    def submit_action(self, action: game_pb2.Action) -> bool:
        """
        Submit an action to the game.
        
        Args:
            action: The action to submit
            
        Returns:
            True if the action was accepted
        """
        if not self.client or not self.credentials:
            raise RuntimeError("Not connected to a game")
        
        try:
            accepted = self.client.submit_action(self.credentials, action)
            if not accepted:
                self.logger.warning(f"Action rejected: {action}")
            return accepted
        except ActionError as e:
            self.logger.error(f"Action error: {e}")
            return False
    
    def _run_polling(self):
        """Run the polling loop in a background thread."""
        try:
            # Define a simple callback that does nothing
            # The real work happens in the event handlers
            def state_callback(state):
                pass
            
            self.session.poll_updates(state_callback)
        except Exception as e:
            self.logger.error(f"Polling error: {e}", exc_info=True)
            self._running = False
    
    def get_current_state(self) -> Optional[game_pb2.GameState]:
        """Get the current game state if available."""
        if self.session and self.session.context.current_game_state:
            return self.session.context.current_game_state
        return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.disconnect()
        return False