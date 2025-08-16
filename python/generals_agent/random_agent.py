"""
New RandomAgent implementation using the refactored architecture.
This agent makes random valid moves demonstrating the simplified agent interface.
"""
import random
import logging
from typing import Optional, List

from .base_agent import BaseAgent
from .types import BoardView, Move, Position
from .game_utils import get_valid_moves, count_player_stats, find_general_position

# Import protobuf types
from generals_pb.game.v1 import game_pb2


class RandomAgent(BaseAgent):
    """
    Random agent implementation using the new architecture.
    
    This agent randomly selects from valid moves, demonstrating how simple
    agent implementation becomes with the new architecture.
    """
    
    def __init__(self, name: str = "RandomAgent"):
        """Initialize the random agent."""
        super().__init__(name)
        self.turn_count = 0
        self.move_count = 0
        self.general_position: Optional[Position] = None
    
    def select_action(self, game_state: game_pb2.GameState) -> Optional[game_pb2.Action]:
        """
        Select a random valid move from the current game state.
        
        Args:
            game_state: Current game state
            
        Returns:
            A random valid action, or None if no valid moves
        """
        self.turn_count += 1
        
        # Get all valid moves for our player
        valid_moves = get_valid_moves(game_state, self.player_id)
        
        if not valid_moves:
            self.logger.debug(f"Turn {self.turn_count}: No valid moves available")
            return None
        
        # Select a random move
        action = random.choice(valid_moves)
        self.move_count += 1
        
        # Log the move
        move_type = "half" if action.half else "full"
        from_coord = getattr(action, 'from')
        to_coord = action.to
        self.logger.info(
            f"Turn {self.turn_count}: {move_type} move from ({from_coord.x},{from_coord.y}) "
            f"to ({to_coord.x},{to_coord.y})"
        )
        
        return action
    
    def select_action_from_board_view(self, board_view: BoardView) -> Optional[Move]:
        """
        Alternative implementation using the simplified BoardView.
        
        This method demonstrates how agents can work with the simplified
        board representation instead of raw protobuf messages.
        
        Args:
            board_view: Simplified board view
            
        Returns:
            A random valid move, or None if no valid moves
        """
        # Get all tiles we own with armies
        owned_tiles = [
            tile for tile in board_view.get_owned_tiles()
            if tile.army_size > 1  # Can only move if we have more than 1 army
        ]
        
        if not owned_tiles:
            return None
        
        # Try random tiles until we find a valid move
        random.shuffle(owned_tiles)
        
        for tile in owned_tiles:
            # Get adjacent positions
            adjacent_positions = [
                Position(tile.position.x + dx, tile.position.y + dy)
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
            ]
            
            # Filter valid adjacent positions
            valid_targets = []
            for pos in adjacent_positions:
                # Check bounds
                if 0 <= pos.x < board_view.width and 0 <= pos.y < board_view.height:
                    target_tile = board_view.get_tile(pos.x, pos.y)
                    if target_tile and target_tile.tile_type != "mountain":
                        valid_targets.append(pos)
            
            if valid_targets:
                # Make a random move
                target = random.choice(valid_targets)
                split = random.choice([True, False])  # Randomly decide to split
                return Move(
                    from_pos=tile.position,
                    to_pos=target,
                    split=split
                )
        
        return None
    
    def on_game_start(self, initial_state: game_pb2.GameState):
        """
        Called when the game starts.
        
        Args:
            initial_state: The initial game state
        """
        self.logger.info(f"Game started! Playing as player {self.player_id}")
        self.turn_count = 0
        self.move_count = 0
        
        # Find our general and log initial stats
        stats = count_player_stats(initial_state, self.player_id)
        general_pos = find_general_position(initial_state, self.player_id)
        
        if general_pos:
            self.general_position = Position(general_pos[0], general_pos[1])
            self.logger.info(
                f"Starting position: General at ({general_pos[0]},{general_pos[1]}), "
                f"{stats['tiles']} tiles, {stats['armies']} armies"
            )
        else:
            self.logger.warning("Could not find general position!")
    
    def on_game_end(self, final_state: game_pb2.GameState, winner_id: str):
        """
        Called when the game ends.
        
        Args:
            final_state: The final game state
            winner_id: ID of the winning player
        """
        if winner_id == self.player_id:
            self.logger.info(f"🎉 Victory! Won in {self.turn_count} turns with {self.move_count} moves")
        else:
            self.logger.info(
                f"Defeat. Lost after {self.turn_count} turns with {self.move_count} moves. "
                f"Winner: {winner_id}"
            )
        
        # Log final statistics
        stats = count_player_stats(final_state, self.player_id)
        self.logger.info(
            f"Final stats: {stats['tiles']} tiles, {stats['armies']} armies, "
            f"{stats['cities']} cities"
        )
    
    def on_turn_start(self, turn: int):
        """
        Called at the start of each turn.
        
        Args:
            turn: Current turn number
        """
        # Log periodic statistics
        if turn % 50 == 0 and turn > 0:
            self.logger.info(f"Reached turn {turn}")
    
    def on_error(self, error: Exception):
        """
        Called when an error occurs.
        
        Args:
            error: The exception that occurred
        """
        self.logger.error(f"Error in RandomAgent: {error}", exc_info=True)


def create_random_agent(name: str = "RandomAgent") -> RandomAgent:
    """
    Factory function to create a random agent.
    
    Args:
        name: Name for the agent
        
    Returns:
        A new RandomAgent instance
    """
    return RandomAgent(name)