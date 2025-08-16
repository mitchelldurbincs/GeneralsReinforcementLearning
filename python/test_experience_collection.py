#!/usr/bin/env python3
"""
End-to-end test of experience collection.
This script:
1. Starts a game with experience collection enabled
2. Runs two agents playing the game
3. Streams experiences in parallel
4. Verifies experiences are being collected
"""

import sys
import os
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from generals_agent import RandomAgent, AgentRunner, GameConfig, GameConnection, GameClient
from generals_agent.experience_consumer import ExperienceConsumer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_agent(agent_name: str, game_id: str, server_address: str):
    """Run a single agent in a thread"""
    try:
        agent = RandomAgent(name=agent_name)
        runner = AgentRunner(agent, server_address=server_address)
        runner.join_game(game_id)
        logging.info(f"{agent_name} joined game {game_id}")
        runner.run()
        logging.info(f"{agent_name} finished game")
    except Exception as e:
        logging.error(f"{agent_name} error: {e}")
        import traceback
        traceback.print_exc()

def stream_experiences(game_id: str, server_address: str, stop_event: threading.Event):
    """Stream experiences from the game"""
    experiences_count = 0
    try:
        with ExperienceConsumer(server_address) as consumer:
            logging.info(f"Starting experience stream for game {game_id}")
            
            # Stream experiences for this specific game
            for batch in consumer.stream_experiences(
                game_ids=[game_id],
                batch_size=10,
                follow=True  # Keep following new experiences
            ):
                for exp in batch:
                    experiences_count += 1
                    if experiences_count % 10 == 0:
                        logging.info(f"Collected {experiences_count} experiences so far")
                    
                    # Log details of first few experiences
                    if experiences_count <= 3:
                        logging.info(f"Experience {experiences_count}:")
                        logging.info(f"  Game: {exp.metadata.get('game_id', 'unknown')}")
                        logging.info(f"  Player: {exp.metadata.get('player_id', 'unknown')}")
                        logging.info(f"  Turn: {exp.metadata.get('turn', 'unknown')}")
                        logging.info(f"  Reward: {exp.reward:.3f}")
                        logging.info(f"  Action: {exp.action}")
                        logging.info(f"  Done: {exp.done}")
                
                # Check if we should stop
                if stop_event.is_set():
                    break
                    
    except Exception as e:
        logging.error(f"Experience streaming error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logging.info(f"Experience stream ended. Total experiences collected: {experiences_count}")

def main():
    """Test experience collection end-to-end"""
    server_address = "localhost:50051"
    
    logging.info("=== Starting Experience Collection Test ===")
    
    # Create game with experience collection enabled
    connection = GameConnection(server_address)
    client = GameClient(connection)
    
    # Create game config with experience collection
    config = GameConfig(
        width=10,
        height=10,
        max_players=2,
        collect_experiences=True  # Enable experience collection
    )
    
    game_id = client.create_game(config)
    logging.info(f"Created game with ID: {game_id}")
    logging.info("Experience collection is ENABLED for this game")
    
    # Check initial experience stats
    try:
        with ExperienceConsumer(server_address) as consumer:
            stats = consumer.get_experience_stats([game_id])
            logging.info(f"Initial stats - Total experiences: {stats.total_experiences}, Games: {stats.total_games}")
    except Exception as e:
        logging.warning(f"Could not get initial stats: {e}")
    
    # Create stop event for experience streaming
    stop_event = threading.Event()
    
    # Start experience streaming in background
    experience_thread = threading.Thread(
        target=stream_experiences,
        args=(game_id, server_address, stop_event)
    )
    experience_thread.start()
    
    # Give streaming a moment to start
    time.sleep(1)
    
    # Run two agents to play the game
    with ThreadPoolExecutor(max_workers=2) as executor:
        agent1_future = executor.submit(run_agent, "Agent1", game_id, server_address)
        agent2_future = executor.submit(run_agent, "Agent2", game_id, server_address)
        
        # Wait for both agents to complete
        agent1_future.result()
        agent2_future.result()
    
    logging.info("Game completed, waiting for final experiences...")
    
    # Give some time for final experiences to be collected
    time.sleep(3)
    
    # Stop experience streaming
    stop_event.set()
    experience_thread.join(timeout=5)
    
    # Get final stats
    try:
        with ExperienceConsumer(server_address) as consumer:
            stats = consumer.get_experience_stats([game_id])
            logging.info(f"Final stats - Total experiences: {stats.total_experiences}, Games: {stats.total_games}")
            
            if stats.total_experiences > 0:
                logging.info("✅ SUCCESS: Experiences were collected!")
                logging.info(f"   Collected {stats.total_experiences} experiences from {stats.total_games} game(s)")
            else:
                logging.warning("⚠️ WARNING: No experiences were collected. Check if:")
                logging.warning("   1. Experience collection is enabled in game config")
                logging.warning("   2. Experience service is running")
                logging.warning("   3. Agents are submitting actions")
    except Exception as e:
        logging.error(f"Could not get final stats: {e}")
    
    logging.info("=== Test Complete ===")

if __name__ == "__main__":
    main()