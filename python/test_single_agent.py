#!/usr/bin/env python3
"""
Test a single random agent with the new architecture.

NOTE: This test has been updated to work with the new agent architecture.
For the recommended way to run agents, see scripts/run_random_match.py
"""

import logging

from generals_agent import (
    RandomAgent, AgentRunner, GameConfig,
    FixedIntervalPolling
)

# Set up very verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run a single agent using the new architecture."""
    # Create agent
    agent = RandomAgent(name="TestAgent")
    
    # Create runner with the agent
    runner = AgentRunner(
        agent,
        server_address="localhost:50051",
        polling_strategy=FixedIntervalPolling(interval=0.5)
    )
    
    try:
        # Create and join a new game
        config = GameConfig(width=10, height=10, max_players=2)
        game_id, player_id = runner.create_and_join_game(config)
        print(f"Created game: {game_id}")
        print(f"Joined as player: {player_id}")
        
        # Note: In a real scenario, you'd need another agent to join
        # before the game can start. This test will likely timeout
        # waiting for another player.
        print("\nWaiting for another player to join...")
        print("(This will timeout - run another agent to actually play)")
        
        # Run the game (will block until game ends)
        runner.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        runner.disconnect()
        print("Disconnected from server")

if __name__ == "__main__":
    main()