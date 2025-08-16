"""
Adapter to make the old BaseAgent compatible with the new architecture.
This allows gradual migration of existing agents.
"""
import logging
from typing import Optional, Dict, Any

from .base_agent import BaseAgent
from .connection import GameConnection
from .game_client import GameClient, PlayerCredentials, GameConfig
from .agent_runner import AgentRunner
from .types import BoardView

# Import protobuf types
from generals_pb.game.v1 import game_pb2


class BaseAgentAdapter(BaseAgent):
    """
    Adapter that wraps the old BaseAgent to work with the new architecture.
    
    This allows existing agents that inherit from the old BaseAgent to work
    with the new AgentRunner without modification.
    """
    
    def __init__(self, old_agent: BaseAgent):
        """
        Initialize the adapter.
        
        Args:
            old_agent: An instance of the old BaseAgent or its subclass
        """
        super().__init__(name=old_agent.agent_name)
        self.old_agent = old_agent
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{old_agent.agent_name}]")
    
    def set_player_id(self, player_id: str):
        """Set player ID on both adapter and old agent."""
        super().set_player_id(player_id)
        self.old_agent.player_id = player_id
    
    def select_action(self, game_state: game_pb2.GameState) -> Optional[game_pb2.Action]:
        """
        Delegate action selection to the old agent.
        
        The old agent expects to be called through its own polling loop,
        so we simulate that behavior here.
        """
        # Update the old agent's state
        self.old_agent.current_game_state = game_state
        
        # Call the old agent's select_action method
        # Note: The old BaseAgent has this as an abstract method
        if hasattr(self.old_agent, 'select_action'):
            return self.old_agent.select_action(game_state)
        else:
            # For agents that don't implement select_action directly,
            # we would need to extract the action from their run loop
            # This is a limitation of the adapter pattern
            self.logger.warning("Old agent doesn't implement select_action directly")
            return None
    
    def on_game_start(self, initial_state: game_pb2.GameState):
        """Notify old agent of game start."""
        self.old_agent.current_game_state = initial_state
        self.logger.info("Game started (adapter)")
    
    def on_game_end(self, final_state: game_pb2.GameState, winner_id: str):
        """Notify old agent of game end."""
        self.old_agent.current_game_state = final_state
        self.logger.info(f"Game ended. Winner: {winner_id} (adapter)")
    
    def on_error(self, error: Exception):
        """Handle errors."""
        self.logger.error(f"Error in adapter: {error}")
        super().on_error(error)


def run_old_agent_with_new_architecture(
    old_agent_class: type[BaseAgent],
    server_address: str = "localhost:50051",
    game_config: Optional[GameConfig] = None,
    join_game_id: Optional[str] = None,
    **agent_kwargs
) -> None:
    """
    Helper function to run an old-style agent with the new architecture.
    
    Args:
        old_agent_class: The old agent class (not instance)
        server_address: Game server address
        game_config: Config for creating a new game (if not joining)
        join_game_id: ID of game to join (if not creating)
        **agent_kwargs: Additional arguments for agent initialization
    """
    # Create old agent instance
    old_agent = old_agent_class(server_address=server_address, **agent_kwargs)
    
    # Wrap in adapter
    adapter = BaseAgentAdapter(old_agent)
    
    # Create runner with new architecture
    runner = AgentRunner(adapter, server_address=server_address)
    
    try:
        # Connect and join/create game
        if join_game_id:
            runner.join_game(join_game_id)
        else:
            config = game_config or GameConfig()
            runner.create_and_join_game(config)
        
        # Run the game
        runner.run()
        
    finally:
        runner.disconnect()


# Monkey patch to add new methods to old BaseAgent for compatibility
def _add_compatibility_methods():
    """Add methods to old BaseAgent for better compatibility."""
    
    def select_action_new(self, game_state: game_pb2.GameState) -> Optional[game_pb2.Action]:
        """
        New select_action method for old agents.
        This attempts to extract action selection logic.
        """
        # This is a placeholder - individual agents would need to override this
        return None
    
    # Only add if not already present
    if not hasattr(BaseAgent, 'select_action'):
        BaseAgent.select_action = select_action_new


# Apply compatibility patches when module is imported
_add_compatibility_methods()