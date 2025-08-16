"""
GameClient class for handling game-specific RPC calls.
Provides a clean interface for interacting with the game server.
"""
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Import the generated gRPC stubs and messages
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))
from generals_pb.game.v1 import game_pb2 as generals_pb2
from generals_pb.game.v1 import game_pb2_grpc as generals_pb2_grpc

from .connection import GameConnection, with_connection_retry


@dataclass
class PlayerCredentials:
    """Credentials for a player in a game."""
    game_id: str
    player_id: str
    player_token: str


@dataclass
class GameConfig:
    """Configuration for creating a new game."""
    width: int = 20
    height: int = 20
    fog_of_war: bool = True
    max_players: int = 2
    turn_time_ms: int = 500  # Time per turn in milliseconds (0 = no limit)
    collect_experiences: bool = False  # Enable experience collection for RL training


class GameClient:
    """Handles game-specific RPC calls to the game server."""
    
    def __init__(self, connection: GameConnection):
        """
        Initialize the game client.
        
        Args:
            connection: The GameConnection instance to use for RPC calls
        """
        self.connection = connection
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @with_connection_retry
    def create_game(self, config: GameConfig) -> str:
        """
        Create a new game on the server.
        
        Args:
            config: Game configuration parameters
            
        Returns:
            The game ID of the created game
            
        Raises:
            GameError: If game creation fails
        """
        stub = self.connection.ensure_connected()
        
        request = generals_pb2.CreateGameRequest(
            config=generals_pb2.GameConfig(
                width=config.width,
                height=config.height,
                fog_of_war=config.fog_of_war,
                max_players=config.max_players,
                turn_time_ms=config.turn_time_ms,
                collect_experiences=config.collect_experiences
            )
        )
        
        try:
            response = stub.CreateGame(request)
            self.logger.info(f"Created game with ID: {response.game_id}")
            return response.game_id
        except grpc.RpcError as e:
            self.logger.error(f"Failed to create game: {e.details()}")
            raise
    
    @with_connection_retry
    def join_game(self, game_id: str, player_name: str) -> PlayerCredentials:
        """
        Join an existing game.
        
        Args:
            game_id: The ID of the game to join
            player_name: The name of the player
            
        Returns:
            PlayerCredentials containing player ID and authentication token
            
        Raises:
            GameError: If joining the game fails
        """
        stub = self.connection.ensure_connected()
        
        request = generals_pb2.JoinGameRequest(
            game_id=game_id,
            player_name=player_name
        )
        
        try:
            response = stub.JoinGame(request)
            self.logger.info(f"Joined game {game_id} as player {response.player_id}")
            return PlayerCredentials(
                game_id=game_id,
                player_id=response.player_id,
                player_token=response.player_token
            )
        except grpc.RpcError as e:
            self.logger.error(f"Failed to join game: {e.details()}")
            raise
    
    @with_connection_retry
    def get_game_state(self, credentials: PlayerCredentials) -> generals_pb2.GameState:
        """
        Get the current game state.
        
        Args:
            credentials: Player credentials for authentication
            
        Returns:
            The current game state
            
        Raises:
            GameError: If fetching game state fails
        """
        stub = self.connection.ensure_connected()
        
        request = generals_pb2.GetGameStateRequest(
            game_id=credentials.game_id,
            player_id=credentials.player_id,
            player_token=credentials.player_token
        )
        
        try:
            response = stub.GetGameState(request)
            return response.state
        except grpc.RpcError as e:
            self.logger.error(f"Failed to get game state: {e.details()}")
            raise
    
    @with_connection_retry
    def submit_action(self, credentials: PlayerCredentials, action: generals_pb2.Action) -> bool:
        """
        Submit an action to the game.
        
        Args:
            credentials: Player credentials for authentication
            action: The action to submit
            
        Returns:
            True if the action was accepted, False otherwise
            
        Raises:
            GameError: If submitting the action fails
        """
        stub = self.connection.ensure_connected()
        
        request = generals_pb2.SubmitActionRequest(
            game_id=credentials.game_id,
            player_id=credentials.player_id,
            player_token=credentials.player_token,
            action=action
        )
        
        try:
            response = stub.SubmitAction(request)
            if response.accepted:
                self.logger.debug(f"Action accepted: {action}")
            else:
                self.logger.warning(f"Action rejected: {action} - Reason: {response.rejection_reason}")
            return response.accepted
        except grpc.RpcError as e:
            self.logger.error(f"Failed to submit action: {e.details()}")
            raise
    
    # Note: LeaveGame and GetAvailableGames are not implemented in the current protobuf
    # These methods are commented out until the server implements them
    
    # @with_connection_retry
    # def leave_game(self, credentials: PlayerCredentials) -> None:
    #     """Leave the current game."""
    #     pass
    
    # @with_connection_retry
    # def get_available_games(self) -> List[generals_pb2.GameInfo]:
    #     """Get a list of available games to join."""
    #     pass