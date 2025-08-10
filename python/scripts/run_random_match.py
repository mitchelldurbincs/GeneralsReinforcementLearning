#!/usr/bin/env python3
"""
Script to run random agent matches using the new architecture.
Demonstrates how much simpler match orchestration becomes with the new design.
"""
import argparse
import logging
import asyncio
import time
from typing import Optional, Dict, List, Tuple
import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generals_agent import (
    RandomAgent, AgentRunner, GameConfig, 
    GameConnection, GameClient, ExponentialBackoffPolling
)


class NewMatchRunner:
    """Manages running matches between agents using the new architecture."""
    
    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.logger = logging.getLogger("NewMatchRunner")
        self.connection = GameConnection(server_address)
        self.client = GameClient(self.connection)
    
    def create_game(self, config: GameConfig) -> Optional[str]:
        """Create a new game on the server."""
        try:
            game_id = self.client.create_game(config)
            self.logger.info(f"Created game with ID: {game_id}")
            return game_id
        except Exception as e:
            self.logger.error(f"Failed to create game: {e}")
            return None
    
    def run_agent(self, agent_name: str, game_id: str) -> Dict:
        """Run a single agent and return results."""
        try:
            # Create agent and runner
            agent = RandomAgent(name=agent_name)
            runner = AgentRunner(
                agent, 
                server_address=self.server_address,
                polling_strategy=ExponentialBackoffPolling(base_interval=0.05),
                enable_logging=False  # Reduce log spam for tournaments
            )
            
            # Join the game
            runner.join_game(game_id)
            
            # Run the agent
            start_time = time.time()
            runner.run(wait_for_players=2, timeout=300)
            duration = time.time() - start_time
            
            # Get final stats
            final_state = runner.get_current_state()
            won = False
            if final_state and runner.session:
                # Check if we won
                for player in final_state.players:
                    if player.id == agent.player_id and player.is_active:
                        # If we're active at game end, we likely won
                        won = True
            
            return {
                'agent_name': agent_name,
                'success': True,
                'won': won,
                'duration': duration,
                'turns': agent.turn_count,
                'moves': agent.move_count
            }
            
        except Exception as e:
            self.logger.error(f"Agent {agent_name} failed: {e}")
            return {
                'agent_name': agent_name,
                'success': False,
                'error': str(e)
            }
        finally:
            if 'runner' in locals():
                runner.disconnect()
    
    def run_match(self, config: GameConfig) -> Dict:
        """Run a single match between two random agents."""
        # Create game
        game_id = self.create_game(config)
        if not game_id:
            return {'success': False, 'error': 'Failed to create game'}
        
        self.logger.info(f"Starting match {game_id}")
        
        # Run both agents concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            agent1_future = executor.submit(self.run_agent, "RandomAgent1", game_id)
            agent2_future = executor.submit(self.run_agent, "RandomAgent2", game_id)
            
            # Wait for both to complete
            agent1_result = agent1_future.result()
            agent2_result = agent2_future.result()
        
        # Determine winner
        winner = None
        if agent1_result['success'] and agent1_result.get('won'):
            winner = agent1_result['agent_name']
        elif agent2_result['success'] and agent2_result.get('won'):
            winner = agent2_result['agent_name']
        
        match_duration = max(
            agent1_result.get('duration', 0),
            agent2_result.get('duration', 0)
        )
        
        return {
            'success': True,
            'game_id': game_id,
            'duration': match_duration,
            'winner': winner,
            'agents': [agent1_result, agent2_result]
        }
    
    def run_tournament(self, num_games: int, config: GameConfig) -> List[Dict]:
        """Run multiple matches and collect statistics."""
        results = []
        wins = {'RandomAgent1': 0, 'RandomAgent2': 0, 'draws': 0}
        
        for i in range(num_games):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting game {i+1}/{num_games}")
            self.logger.info(f"{'='*50}")
            
            match_result = self.run_match(config)
            results.append(match_result)
            
            if match_result['success']:
                winner = match_result.get('winner')
                if winner:
                    wins[winner] += 1
                else:
                    wins['draws'] += 1
            
            # Log current standings
            self.logger.info(f"Current standings: Agent1={wins['RandomAgent1']}, "
                           f"Agent2={wins['RandomAgent2']}, Draws={wins['draws']}")
            
            # Small delay between games
            time.sleep(1)
        
        return results, wins
    
    def cleanup(self):
        """Clean up resources."""
        self.connection.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run matches between random Generals.io agents using new architecture'
    )
    parser.add_argument('--server', default='localhost:50051', 
                       help='Server address (default: localhost:50051)')
    parser.add_argument('--games', type=int, default=1,
                       help='Number of games to play (default: 1)')
    parser.add_argument('--width', type=int, default=20,
                       help='Board width (default: 20)')
    parser.add_argument('--height', type=int, default=20,
                       help='Board height (default: 20)')
    parser.add_argument('--no-fog', action='store_true',
                       help='Disable fog of war')
    parser.add_argument('--players', type=int, default=2,
                       help='Number of players (default: 2)')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create game configuration
    game_config = GameConfig(
        width=args.width,
        height=args.height,
        fog_of_war_enabled=not args.no_fog,
        min_players=args.players,
        max_players=args.players
    )
    
    # Create match runner
    runner = NewMatchRunner(server_address=args.server)
    
    try:
        if args.games == 1:
            # Single match
            result = runner.run_match(game_config)
            
            if result['success']:
                print(f"\nMatch completed!")
                print(f"Game ID: {result['game_id']}")
                print(f"Duration: {result['duration']:.1f} seconds")
                print(f"Winner: {result.get('winner', 'Draw')}")
                
                # Print agent details
                for agent_result in result['agents']:
                    print(f"\n{agent_result['agent_name']}:")
                    if agent_result['success']:
                        print(f"  Turns: {agent_result['turns']}")
                        print(f"  Moves: {agent_result['moves']}")
                    else:
                        print(f"  Failed: {agent_result.get('error', 'Unknown error')}")
            else:
                print(f"\nMatch failed: {result.get('error', 'Unknown error')}")
        
        else:
            # Tournament
            results, wins = runner.run_tournament(args.games, game_config)
            
            # Calculate statistics
            total_games = len(results)
            successful_games = sum(1 for r in results if r['success'])
            total_duration = sum(r['duration'] for r in results if r['success'])
            
            # Print summary
            print(f"\n{'='*50}")
            print("TOURNAMENT SUMMARY")
            print(f"{'='*50}")
            print(f"Total games: {total_games}")
            print(f"Successful games: {successful_games}")
            print(f"Failed games: {total_games - successful_games}")
            
            if successful_games > 0:
                print(f"\nResults:")
                print(f"  RandomAgent1 wins: {wins['RandomAgent1']} "
                      f"({wins['RandomAgent1']/successful_games*100:.1f}%)")
                print(f"  RandomAgent2 wins: {wins['RandomAgent2']} "
                      f"({wins['RandomAgent2']/successful_games*100:.1f}%)")
                print(f"  Draws: {wins['draws']} "
                      f"({wins['draws']/successful_games*100:.1f}%)")
                print(f"\nAverage game duration: {total_duration/successful_games:.1f} seconds")
            
            # Save results if requested
            if args.output:
                output_data = {
                    'config': {
                        'num_games': args.games,
                        'width': args.width,
                        'height': args.height,
                        'fog_of_war': not args.no_fog,
                        'players': args.players
                    },
                    'summary': {
                        'total_games': total_games,
                        'successful_games': successful_games,
                        'wins': wins,
                        'avg_duration': total_duration/successful_games if successful_games > 0 else 0
                    },
                    'matches': results
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to {args.output}")
    
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()