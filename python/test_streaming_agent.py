#!/usr/bin/env python3
"""Test script to verify StreamGame functionality with random agents"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import time
import threading
from generals_agent.random_agent import RandomAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_agent(agent_name: str, game_id: str, use_streaming: bool):
    """Run a single agent in a thread"""
    try:
        agent = RandomAgent(agent_name=agent_name)
        agent.run(game_id=game_id, use_streaming=use_streaming)
    except Exception as e:
        logging.error(f"{agent_name} error: {e}")

def main():
    """Test streaming with two random agents"""
    # Create first agent to create game
    logging.info("Starting streaming test...")
    
    agent1 = RandomAgent(agent_name="StreamingAgent1")
    agent1.connect()
    game_id = agent1.create_game(width=10, height=10)
    agent1.disconnect()
    
    logging.info(f"Created game: {game_id}")
    
    # Start both agents in threads using streaming
    thread1 = threading.Thread(
        target=run_agent, 
        args=("StreamingAgent1", game_id, True),
        name="Agent1"
    )
    thread2 = threading.Thread(
        target=run_agent, 
        args=("StreamingAgent2", game_id, True),
        name="Agent2"
    )
    
    logging.info("Starting agents with streaming enabled...")
    thread1.start()
    time.sleep(0.5)  # Small delay before starting second agent
    thread2.start()
    
    # Wait for both to complete
    thread1.join(timeout=60)
    thread2.join(timeout=60)
    
    if thread1.is_alive() or thread2.is_alive():
        logging.error("Game took too long, threads still running")
    else:
        logging.info("Game completed successfully with streaming!")

if __name__ == "__main__":
    main()