#!/usr/bin/env python3
"""Script to run random agent matches"""

import argparse
import logging
import multiprocessing
import time
import grpc
from typing import Optional, Dict, List
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_agent.random_agent import RandomAgent


class MatchRunner:
    """Manages running matches between agents"""
    
    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.logger = logging.getLogger("MatchRunner")
        
    def create_game(self, width: int = 20, height: int = 20, 
                   fog_of_war: bool = True) -> Optional[str]:
        """Create a new game on the server"""
        try:
            channel = grpc.insecure_channel(self.server_address)
            stub = game_pb2_grpc.GameServiceStub(channel)
            
            request = game_pb2.CreateGameRequest(
                config=game_pb2.GameConfig(
                    width=width,
                    height=height,
                    fog_of_war=fog_of_war,
                    max_players=2
                )
            )
            
            response = stub.CreateGame(request)
            game_id = response.game_id
            self.logger.info(f"Created game with ID: {game_id}")
            channel.close()
            return game_id
            
        except grpc.RpcError as e:
            self.logger.error(f"Failed to create game: {e}")
            return None
            
    def run_agent_process(self, agent_name: str, game_id: str, 
                         results_queue: multiprocessing.Queue):
        """Run an agent in a separate process"""
        # Set up logging for subprocess
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - {agent_name} - %(levelname)s - %(message)s'
        )
        
        try:
            agent = RandomAgent(server_address=self.server_address, 
                              agent_name=agent_name)
            agent.run(game_id=game_id)
            
            # Report results
            results_queue.put({
                'agent_name': agent_name,
                'success': True,
                'turn_count': agent.turn_count,
                'move_count': agent.move_count
            })
            
        except Exception as e:
            results_queue.put({
                'agent_name': agent_name,
                'success': False,
                'error': str(e)
            })
            
    def run_match(self, width: int = 20, height: int = 20, 
                  fog_of_war: bool = True) -> Dict:
        """Run a single match between two random agents"""
        # Create game
        game_id = self.create_game(width, height, fog_of_war)
        if not game_id:
            return {'success': False, 'error': 'Failed to create game'}
            
        # Set up results queue
        results_queue = multiprocessing.Queue()
        
        # Start two agent processes
        agent1 = multiprocessing.Process(
            target=self.run_agent_process,
            args=("RandomAgent1", game_id, results_queue)
        )
        agent2 = multiprocessing.Process(
            target=self.run_agent_process,
            args=("RandomAgent2", game_id, results_queue)
        )
        
        self.logger.info(f"Starting match {game_id}")
        start_time = time.time()
        
        agent1.start()
        time.sleep(0.5)  # Small delay to avoid race conditions
        agent2.start()
        
        # Wait for both agents to finish
        agent1.join(timeout=300)  # 5 minute timeout
        agent2.join(timeout=300)
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
            
        match_time = time.time() - start_time
        
        # Determine winner
        winner = None
        for result in results:
            if result.get('success'):
                # Agent that succeeded is likely the winner
                # (The other would have disconnected on game over)
                winner = result['agent_name']
                
        return {
            'success': True,
            'game_id': game_id,
            'duration': match_time,
            'results': results,
            'winner': winner
        }
        
    def run_tournament(self, num_games: int, **game_config) -> List[Dict]:
        """Run multiple matches and collect statistics"""
        results = []
        wins = {'RandomAgent1': 0, 'RandomAgent2': 0}
        
        for i in range(num_games):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting game {i+1}/{num_games}")
            self.logger.info(f"{'='*50}")
            
            match_result = self.run_match(**game_config)
            results.append(match_result)
            
            if match_result['success'] and match_result.get('winner'):
                wins[match_result['winner']] += 1
                
            # Log current standings
            self.logger.info(f"Current standings: {wins}")
            
            # Small delay between games
            time.sleep(1)
            
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run matches between random Generals.io agents'
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
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create match runner
    runner = MatchRunner(server_address=args.server)
    
    # Run matches
    if args.games == 1:
        # Single match
        result = runner.run_match(
            width=args.width,
            height=args.height,
            fog_of_war=not args.no_fog
        )
        
        if result['success']:
            print(f"\nMatch completed!")
            print(f"Game ID: {result['game_id']}")
            print(f"Duration: {result['duration']:.1f} seconds")
            print(f"Winner: {result.get('winner', 'Unknown')}")
        else:
            print(f"\nMatch failed: {result.get('error', 'Unknown error')}")
            
    else:
        # Tournament
        results = runner.run_tournament(
            num_games=args.games,
            width=args.width,
            height=args.height,
            fog_of_war=not args.no_fog
        )
        
        # Calculate statistics
        total_games = len(results)
        successful_games = sum(1 for r in results if r['success'])
        wins = {'RandomAgent1': 0, 'RandomAgent2': 0}
        total_duration = 0
        
        for result in results:
            if result['success']:
                total_duration += result['duration']
                if result.get('winner'):
                    wins[result['winner']] += 1
                    
        # Print summary
        print(f"\n{'='*50}")
        print("TOURNAMENT SUMMARY")
        print(f"{'='*50}")
        print(f"Total games: {total_games}")
        print(f"Successful games: {successful_games}")
        print(f"Failed games: {total_games - successful_games}")
        print(f"\nWins:")
        print(f"  RandomAgent1: {wins['RandomAgent1']} ({wins['RandomAgent1']/successful_games*100:.1f}%)")
        print(f"  RandomAgent2: {wins['RandomAgent2']} ({wins['RandomAgent2']/successful_games*100:.1f}%)")
        print(f"\nAverage game duration: {total_duration/successful_games:.1f} seconds")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'config': {
                        'num_games': args.games,
                        'width': args.width,
                        'height': args.height,
                        'fog_of_war': not args.no_fog
                    },
                    'summary': {
                        'total_games': total_games,
                        'successful_games': successful_games,
                        'wins': wins,
                        'avg_duration': total_duration/successful_games if successful_games > 0 else 0
                    },
                    'matches': results
                }, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()