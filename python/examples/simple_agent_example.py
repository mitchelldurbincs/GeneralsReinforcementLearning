#!/usr/bin/env python3
"""
Simple example demonstrating how to create and run an agent with the new architecture.
This shows how much simpler agent development becomes.
"""
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generals_agent import (
    BaseAgentNew, AgentRunner, GameConfig,
    BoardView, Move, Position
)
from generals_pb.game.v1 import game_pb2


class SimpleExampleAgent(BaseAgentNew):
    """
    A simple example agent that demonstrates the new architecture.
    
    This agent:
    - Always moves its largest army
    - Prefers to capture enemy tiles
    - Falls back to expanding to neutral tiles
    """
    
    def __init__(self):
        super().__init__(name="SimpleExample")
        self.my_general_pos = None
    
    def select_action_from_board_view(self, board_view: BoardView) -> Optional[Move]:
        """
        Select an action using the simplified board view.
        This is often easier than working with raw protobuf.
        """
        # Find our tiles with moveable armies (> 1)
        moveable_tiles = [
            tile for tile in board_view.get_owned_tiles()
            if tile.army_size > 1
        ]
        
        if not moveable_tiles:
            return None
        
        # Sort by army size (largest first)
        moveable_tiles.sort(key=lambda t: t.army_size, reverse=True)
        
        # Try to find a good move for our largest army
        for tile in moveable_tiles:
            # Get adjacent positions
            adjacent = self._get_adjacent_positions(tile.position, board_view)
            
            # Priority 1: Attack enemy tiles
            enemy_targets = []
            for pos in adjacent:
                target = board_view.get_tile(pos.x, pos.y)
                if target and target.owner_id and target.owner_id != self.player_id:
                    enemy_targets.append((pos, target))
            
            if enemy_targets:
                # Attack the weakest enemy
                enemy_targets.sort(key=lambda x: x[1].army_size)
                target_pos, _ = enemy_targets[0]
                return Move(tile.position, target_pos, split=False)
            
            # Priority 2: Capture neutral tiles
            neutral_targets = []
            for pos in adjacent:
                target = board_view.get_tile(pos.x, pos.y)
                if target and not target.owner_id and target.tile_type != "mountain":
                    neutral_targets.append(pos)
            
            if neutral_targets:
                # Prefer cities
                for pos in neutral_targets:
                    target = board_view.get_tile(pos.x, pos.y)
                    if target.tile_type == "city":
                        return Move(tile.position, pos, split=False)
                
                # Otherwise take any neutral
                return Move(tile.position, neutral_targets[0], split=False)
        
        # If no good targets, just move our largest army somewhere
        largest_army_tile = moveable_tiles[0]
        adjacent = self._get_adjacent_positions(largest_army_tile.position, board_view)
        
        for pos in adjacent:
            target = board_view.get_tile(pos.x, pos.y)
            if target and target.tile_type != "mountain":
                # Move half to maintain presence
                return Move(largest_army_tile.position, pos, split=True)
        
        return None
    
    def _get_adjacent_positions(self, pos: Position, board_view: BoardView) -> List[Position]:
        """Get valid adjacent positions."""
        adjacent = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = pos.x + dx, pos.y + dy
            if 0 <= new_x < board_view.width and 0 <= new_y < board_view.height:
                adjacent.append(Position(new_x, new_y))
        return adjacent
    
    def on_game_start(self, initial_state: game_pb2.GameState):
        """Remember our general's position."""
        self.logger.info("Game started! Looking for our general...")
        
        # Find our general
        for y in range(initial_state.board.height):
            for x in range(initial_state.board.width):
                idx = y * initial_state.board.width + x
                if idx < len(initial_state.board.tiles):
                    tile = initial_state.board.tiles[idx]
                    if (tile.type == game_pb2.TileType.GENERAL and 
                        tile.owner_id == self.player_id):
                        self.my_general_pos = Position(x, y)
                        self.logger.info(f"Found our general at ({x}, {y})")
                        return
    
    def on_game_end(self, final_state: game_pb2.GameState, winner_id: str):
        """Log the game result."""
        if winner_id == self.player_id:
            self.logger.info("🎉 We won!")
        else:
            self.logger.info(f"We lost. Winner: {winner_id}")


def main():
    """Example of running the agent."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create our agent
    agent = SimpleExampleAgent()
    
    # Create the runner that handles all infrastructure
    runner = AgentRunner(agent, server_address="localhost:50051")
    
    # Create a new game
    game_config = GameConfig(
        width=15,
        height=15,
        fog_of_war_enabled=True,
        min_players=2,
        max_players=2
    )
    
    try:
        # Create and join the game
        credentials = runner.create_and_join_game(game_config)
        print(f"Joined game {credentials.game_id} as player {credentials.player_id}")
        
        # Wait for another player and run the game
        print("Waiting for opponent...")
        runner.run(wait_for_players=2)
        
    finally:
        # Clean up
        runner.disconnect()


if __name__ == '__main__':
    main()