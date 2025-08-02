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
        from generals_pb.common.v1 import common_pb2
        
        request = game_pb2.SubmitActionRequest(
            game_id=self.game_id,
            player_id=self.player_id,
            player_token=self.player_token,
            action=action
        )
        
        try:
            response = self.stub.SubmitAction(request)
            if response.success:
                if action.type == common_pb2.ACTION_TYPE_MOVE:
                    from_coord = getattr(action, 'from')
                    self.logger.debug(f"Move action submitted successfully from ({from_coord.x},{from_coord.y}) to ({action.to.x},{action.to.y})")
                else:
                    self.logger.debug(f"Action submitted successfully: type={common_pb2.ActionType.Name(action.type)}")
            else:
                self.logger.warning(f"Action rejected: {response.error_message}")
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
            first_poll = True
            while True:
                # Get current game state
                game_state = self.get_game_state()
                
                # Log first poll details
                if first_poll:
                    status_name = common_pb2.GameStatus.Name(game_state.status)
                    self.logger.info(f"First poll - Game status: {status_name}, Turn: {game_state.turn}, Players: {len(game_state.players)}")
                    first_poll = False
                
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
                
                # Check if we need to process this state
                should_process = False
                
                # Process if turn changed
                if game_state.turn > last_turn:
                    self.logger.debug(f"Turn changed from {last_turn} to {game_state.turn}, status: {common_pb2.GameStatus.Name(game_state.status)}")
                    last_turn = game_state.turn
                    should_process = True
                    
                # Also process if status changed from WAITING to IN_PROGRESS at the same turn
                if not hasattr(self, '_last_status'):
                    self._last_status = game_state.status
                elif self._last_status == common_pb2.GAME_STATUS_WAITING and game_state.status == common_pb2.GAME_STATUS_IN_PROGRESS:
                    self.logger.info(f"Game started! Status changed from WAITING to IN_PROGRESS at turn {game_state.turn}")
                    should_process = True
                    
                self._last_status = game_state.status
                self.current_game_state = game_state
                    
                # Submit actions if we should process and game is in progress
                if should_process and game_state.status == common_pb2.GAME_STATUS_IN_PROGRESS:
                    try:
                        self.on_state_update(game_state)
                    except Exception as e:
                        self.logger.error(f"Error in on_state_update: {e}")
                        import traceback
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        raise
                
                # Small delay to avoid hammering the server
                time.sleep(poll_interval)
                
        except grpc.RpcError as e:
            self.logger.error(f"Polling error: {e}")
            self.on_disconnect()
        except Exception as e:
            import traceback
            self.logger.error(f"Unexpected error during polling: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.on_disconnect()
    
    def stream_game_updates(self):
        """Stream game updates from the server using StreamGame"""
        from generals_pb.common.v1 import common_pb2
        
        try:
            self.logger.info("Starting to stream game updates...")
            
            # Create stream request
            stream_request = game_pb2.StreamGameRequest(
                game_id=self.game_id,
                player_id=self.player_id,
                player_token=self.player_token
            )
            
            # Connect to stream
            stream = self.stub.StreamGame(stream_request)
            first_update = True
            
            # Process stream updates
            for update in stream:
                try:
                    # Handle different update types
                    if update.HasField('full_state'):
                        game_state = update.full_state
                        self.current_game_state = game_state
                        
                        if first_update:
                            status_name = common_pb2.GameStatus.Name(game_state.status)
                            self.logger.info(f"First update - Game status: {status_name}, Turn: {game_state.turn}")
                            first_update = False
                        
                        # Check if game ended
                        if game_state.status == common_pb2.GAME_STATUS_FINISHED:
                            winner_id = game_state.winner_id if game_state.winner_id > 0 else None
                            game_ended = game_pb2.GameEndedEvent()
                            if winner_id is not None:
                                game_ended.winner_id = winner_id
                            self.on_game_end(game_ended)
                            break
                        
                        # Submit action if game is in progress
                        if game_state.status == common_pb2.GAME_STATUS_IN_PROGRESS:
                            self.on_state_update(game_state)
                    
                    elif update.HasField('event'):
                        # Handle specific events
                        event = update.event
                        if event.HasField('game_started'):
                            self.logger.info("Game started event received")
                        elif event.HasField('game_ended'):
                            self.on_game_end(event.game_ended)
                            break
                        elif event.HasField('player_eliminated'):
                            self.logger.info(f"Player {event.player_eliminated.player_id} eliminated")
                        elif event.HasField('phase_changed'):
                            phase_event = event.phase_changed
                            self.logger.debug(f"Phase changed: {common_pb2.GamePhase.Name(phase_event.previous_phase)} -> {common_pb2.GamePhase.Name(phase_event.new_phase)}")
                    
                    elif update.HasField('delta'):
                        # Handle delta updates (incremental state changes)
                        self.logger.debug(f"Received delta update for turn {update.delta.turn}")
                        # For now, we'll request full state on next update
                        # In the future, we could apply deltas to local state
                        
                except Exception as e:
                    self.logger.error(f"Error processing stream update: {e}")
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                self.logger.info("Stream cancelled")
            else:
                self.logger.error(f"Streaming error: {e.code()}: {e.details()}")
            self.on_disconnect()
        except Exception as e:
            self.logger.error(f"Unexpected error during streaming: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.on_disconnect()
            
    def run(self, game_id: Optional[str] = None, use_streaming: bool = True):
        """Main game loop"""
        try:
            self.connect()
            
            # Create or join game
            if game_id is None:
                game_id = self.create_game()
            self.join_game(game_id)
            
            # Wait a moment for the game to fully initialize
            time.sleep(0.5)
            
            # Notify game start
            self.on_game_start()
            
            # Main game loop - use streaming or polling
            if use_streaming:
                self.stream_game_updates()
            else:
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
        
        # In Generals, all players submit actions simultaneously for each turn
        action = self.select_action(game_state)
        if action:
            self.submit_action(action)
        else:
            # Submit a no-op action to signal we're ready for the next turn
            self.logger.debug(f"Turn {game_state.turn}: No valid moves, submitting no-op action")
            no_op_action = game_pb2.Action(
                type=common_pb2.ACTION_TYPE_UNSPECIFIED,
                turn_number=game_state.turn
            )
            self.submit_action(no_op_action)
                
    def on_disconnect(self):
        """Called when disconnected from server"""
        self.logger.warning("Disconnected from server")
        
    def wait_for_turn(self):
        """Wait for next turn (all players act simultaneously in Generals)"""
        from generals_pb.common.v1 import common_pb2
        
        # In Generals, all players act simultaneously, so we always return True
        # unless the game has ended
        try:
            state = self.get_game_state()
            return state.status != common_pb2.GAME_STATUS_FINISHED
        except grpc.RpcError:
            return False