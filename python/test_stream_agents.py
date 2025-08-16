#!/usr/bin/env python3
"""Test streaming with two random agents playing"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from generals_agent import RandomAgent, AgentRunner, GameConfig, GameConnection, GameClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_agent(agent_name, game_id, server_address):
    """Run a single agent"""
    try:
        agent = RandomAgent(name=agent_name)
        runner = AgentRunner(agent, server_address=server_address)
        runner.join_game(game_id)
        logging.info(f"{agent_name} joined game {game_id}")
        runner.run()
        logging.info(f"{agent_name} finished game")
    except Exception as e:
        logging.error(f"{agent_name} error: {e}")

def main():
    server_address = "localhost:50051"
    
    # Create game
    connection = GameConnection(server_address)
    client = GameClient(connection)
    
    config = GameConfig(
        width=10,
        height=10,
        max_players=2,
        turn_time_ms=500  # 500ms per turn
    )
    
    game_id = client.create_game(config)
    logging.info(f"Created game: {game_id}")
    
    # Run two agents
    with ThreadPoolExecutor(max_workers=2) as executor:
        agent1_future = executor.submit(run_agent, "Agent1", game_id, server_address)
        agent2_future = executor.submit(run_agent, "Agent2", game_id, server_address)
        
        # Wait for both agents
        agent1_future.result()
        agent2_future.result()
    
    logging.info("Game completed!")

if __name__ == "__main__":
    main()