"""Random Agent implementation for Generals.io"""

import random
import logging
from typing import Optional

from .base_agent import BaseAgent
from .game_utils import get_valid_moves, count_player_stats, find_general_position

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generals_pb.game.v1 import game_pb2


class RandomAgent(BaseAgent):
    """Agent that plays random valid moves"""
    
    def __init__(self, server_address: str = "localhost:50051", agent_name: str = "RandomAgent"):
        super().__init__(server_address, agent_name)
        self.turn_count = 0
        self.move_count = 0
        
    def select_action(self, game_state: game_pb2.GameState) -> Optional[game_pb2.Action]:
        """Select a random valid move"""
        self.turn_count += 1
        
        # Get all valid moves
        valid_moves = get_valid_moves(game_state, self.player_id)
        
        if not valid_moves:
            self.logger.debug(f"Turn {self.turn_count}: No valid moves available")
            return None
            
        # Select random move
        action = random.choice(valid_moves)
        self.move_count += 1
        
        # Log move details
        move_type = "MOVE_HALF" if action.half else "MOVE"
        from_coord = getattr(action, 'from')
        self.logger.info(f"Turn {self.turn_count}: {move_type} from ({from_coord.x},{from_coord.y}) "
                        f"to ({action.to.x},{action.to.y})")
        
        return action
        
    def on_game_start(self):
        """Called when game starts"""
        self.logger.info(f"Game started! Playing as player {self.player_id}")
        self.turn_count = 0
        self.move_count = 0
        
        # Get initial state and log stats
        initial_state = self.get_game_state()
        stats = count_player_stats(initial_state, self.player_id)
        general_pos = find_general_position(initial_state, self.player_id)
        
        self.logger.info(f"Starting position: General at {general_pos}, "
                        f"{stats['tiles']} tiles, {stats['armies']} armies")
        
    def on_game_end(self, game_ended: game_pb2.GameEndedEvent):
        """Called when game ends"""
        if game_ended.winner_id == self.player_id:
            self.logger.info(f"Victory! Won in {self.turn_count} turns with {self.move_count} moves")
        else:
            self.logger.info(f"Defeat. Lost after {self.turn_count} turns with {self.move_count} moves. "
                           f"Winner: Player {game_ended.winner_id}")
            
        # Log final stats if we have the final state
        if self.current_game_state:
            stats = count_player_stats(self.current_game_state, self.player_id)
            self.logger.info(f"Final stats: {stats['tiles']} tiles, {stats['armies']} armies, "
                           f"{stats['cities']} cities")
            
    def on_state_update(self, game_state: game_pb2.GameState):
        """Handle state updates with additional logging"""
        # Log periodic stats
        if self.turn_count % 50 == 0 and self.turn_count > 0:
            stats = count_player_stats(game_state, self.player_id)
            self.logger.info(f"Turn {self.turn_count} stats: {stats['tiles']} tiles, "
                           f"{stats['armies']} armies, {stats['cities']} cities")
            
        # Call parent to handle turn logic
        super().on_state_update(game_state)


def main():
    """Run a single random agent"""
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Run a random Generals.io agent')
    parser.add_argument('--server', default='localhost:50051', help='Server address')
    parser.add_argument('--game-id', help='Game ID to join (creates new game if not specified)')
    parser.add_argument('--name', default='RandomAgent', help='Agent name')
    
    args = parser.parse_args()
    
    # Create and run agent
    agent = RandomAgent(server_address=args.server, agent_name=args.name)
    agent.run(game_id=args.game_id)


if __name__ == '__main__':
    main()