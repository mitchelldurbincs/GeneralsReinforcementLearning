"""Base Agent class for Generals.io agents"""

import grpc
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import logging
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generals_pb.game.v1 import game_pb2
from generals_pb.game.v1 import game_pb2_grpc


class BaseAgent(ABC):
    """Abstract base class for Generals.io agents"""
    
    def __init__(self, server_address: str = "localhost:50051", agent_name: str = "BaseAgent"):
        self.server_address = server_address
        self.agent_name = agent_name
        self.channel = None
        self.stub = None
        self.game_id = None
        self.player_id = None
        self.player_token = None
        self.current_game_state = None
        self.logger = logging.getLogger(agent_name)
        
        # Connection retry settings
        self.max_retries = 5
        self.retry_delay = 1.0  # seconds
        
    def connect(self):
        """Connect to the game server"""
        self.logger.info(f"Connecting to server at {self.server_address}")
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = game_pb2_grpc.GameServiceStub(self.channel)
        
    def disconnect(self):
        """Disconnect from the game server"""
        if self.channel:
            self.channel.close()
            self.logger.info("Disconnected from server")
            
    def create_game(self, width: int = 20, height: int = 20, 
                   fog_of_war: bool = True) -> str:
        """Create a new game and return the game ID"""
        request = game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(
                width=width,
                height=height,
                fog_of_war=fog_of_war,
                max_players=2
            )
        )
        
        response = self.stub.CreateGame(request)
        self.game_id = response.game_id
        self.logger.info(f"Created game with ID: {self.game_id}")
        return self.game_id
        
    def join_game(self, game_id: str) -> Tuple[str, str]:
        """Join a game and return player_id and token"""
        self.game_id = game_id
        request = game_pb2.JoinGameRequest(
            game_id=game_id,
            player_name=self.agent_name
        )
        
        response = self.stub.JoinGame(request)
        self.player_id = response.player_id
        self.player_token = response.player_token
        self.logger.info(f"Joined game {game_id} as player {self.player_id}")
        return self.player_id, self.player_token
        
    def get_game_state(self) -> game_pb2.GameState:
        """Get the current game state"""
        request = game_pb2.GetGameStateRequest(
            game_id=self.game_id,
            player_id=self.player_id,
            player_token=self.player_token
        )
        
        response = self.stub.GetGameState(request)
        self.current_game_state = response.state
        return self.current_game_state
        
    def submit_action(self, action: game_pb2.Action) -> bool:
        """Submit an action to the server"""
        request = game_pb2.SubmitActionRequest(
            game_id=self.game_id,
            player_id=self.player_id,
            player_token=self.player_token,
            action=action
        )
        
        try:
            response = self.stub.SubmitAction(request)
            if response.success:
                self.logger.debug(f"Action submitted successfully: {action}")
            else:
                self.logger.warning(f"Action rejected: {response.message}")
            return response.success
        except grpc.RpcError as e:
            self.logger.error(f"Error submitting action: {e}")
            return False
            
    def poll_game_updates(self):
        """Poll game updates from the server using GetGameState"""
        from generals_pb.common.v1 import common_pb2
        
        poll_interval = 0.1  # seconds
        last_turn = -1
        
        try:
            self.logger.debug("Starting to poll game updates...")
            while True:
                # Get current game state
                game_state = self.get_game_state()
                
                # Check if game has ended
                if game_state.status == common_pb2.GAME_STATUS_FINISHED:
                    # Find the winner - the player who is not eliminated
                    winner_id = None
                    for player in game_state.players:
                        if player.status != common_pb2.PLAYER_STATUS_ELIMINATED:
                            winner_id = player.id
                            break
                    
                    # Create a game ended event
                    game_ended = game_pb2.GameEndedEvent()
                    if winner_id is not None:
                        game_ended.winner_id = winner_id
                    
                    self.on_game_end(game_ended)
                    break
                
                # Process state update if turn has changed
                if game_state.turn > last_turn:
                    last_turn = game_state.turn
                    self.current_game_state = game_state
                    self.on_state_update(game_state)
                
                # Small delay to avoid hammering the server
                time.sleep(poll_interval)
                
        except grpc.RpcError as e:
            self.logger.error(f"Polling error: {e}")
            self.on_disconnect()
        except Exception as e:
            self.logger.error(f"Unexpected error during polling: {e}")
            self.on_disconnect()
            
    def run(self, game_id: Optional[str] = None):
        """Main game loop"""
        try:
            self.connect()
            
            # Create or join game
            if game_id is None:
                game_id = self.create_game()
            self.join_game(game_id)
            
            # Notify game start
            self.on_game_start()
            
            # Main game loop - poll updates and respond
            self.poll_game_updates()
            
        except Exception as e:
            self.logger.error(f"Agent error: {e}")
        finally:
            self.disconnect()
            
    @abstractmethod
    def select_action(self, game_state: game_pb2.GameState) -> Optional[game_pb2.Action]:
        """Select an action based on the current game state"""
        pass
        
    @abstractmethod
    def on_game_start(self):
        """Called when the game starts"""
        pass
        
    @abstractmethod
    def on_game_end(self, game_ended: game_pb2.GameEndedEvent):
        """Called when the game ends"""
        pass
        
    def on_state_update(self, game_state: game_pb2.GameState):
        """Called when game state is updated"""
        # Import here to avoid circular dependency
        from generals_pb.common.v1 import common_pb2
        
        # Only process if game is in progress
        if game_state.status != common_pb2.GAME_STATUS_IN_PROGRESS:
            self.logger.debug(f"Game not in progress, status: {game_state.status}")
            return
            
        # Check if it's our turn
        current_player_turn = game_state.turn % 2  # Assuming 2 players, turn alternates
        if current_player_turn == int(self.player_id):
            action = self.select_action(game_state)
            if action:
                self.submit_action(action)
                
    def on_disconnect(self):
        """Called when disconnected from server"""
        self.logger.warning("Disconnected from server")
        
    def wait_for_turn(self):
        """Wait for our turn by polling game state"""
        from generals_pb.common.v1 import common_pb2
        
        while True:
            try:
                state = self.get_game_state()
                if state.status == common_pb2.GAME_STATUS_FINISHED:
                    return False
                # Check if it's our turn (alternating turns)
                current_player_turn = state.turn % 2
                if current_player_turn == int(self.player_id):
                    return True
                time.sleep(0.1)  # Small delay to avoid hammering the server
            except grpc.RpcError:
                return False