#!/usr/bin/env python3
"""
Test script to verify game functionality with random agents.

NOTE: This test has been updated to work with the new agent architecture.
For the recommended way to run multi-agent matches, see scripts/run_random_match.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from generals_agent import (
    RandomAgent, AgentRunner, GameConfig, 
    GameConnection, GameClient, ExponentialBackoffPolling
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_agent(agent_name: str, game_id: str):
    """Run a single agent in a thread"""
    try:
        # Create agent and runner
        agent = RandomAgent(name=agent_name)
        runner = AgentRunner(
            agent,
            server_address="localhost:50051", 
            polling_strategy=ExponentialBackoffPolling(initial_interval=0.1, max_interval=2.0)
        )
        
        # Join existing game and run
        runner.join_game(game_id)
        logging.info(f"{agent_name} joined game {game_id}")
        runner.run()
        
    except Exception as e:
        logging.error(f"{agent_name} error: {e}")
    finally:
        try:
            runner.disconnect()
        except:
            pass

def main():
    """Test with two random agents"""
    logging.info("Starting agent test...")
    
    # Create game using client
    connection = GameConnection("localhost:50051")
    client = GameClient(connection)
    
    config = GameConfig(width=15, height=15, max_players=2)
    game_id = client.create_game(config)
    logging.info(f"Created game with ID: {game_id}")
    
    # Run two agents concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        agent1_future = executor.submit(run_agent, "Agent1", game_id)
        agent2_future = executor.submit(run_agent, "Agent2", game_id)
        
        # Wait for both to complete
        agent1_future.result()
        agent2_future.result()
    
    logging.info("Test completed!")

if __name__ == "__main__":
    main()