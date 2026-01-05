#!/usr/bin/env python3
"""
Test script to verify end-to-end experience streaming from game server to Python client.
This tests the complete pipeline from game creation to experience consumption.
"""

import sys
import os
import time
import threading
import grpc
import numpy as np
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.experience.v1 import experience_pb2, experience_pb2_grpc
from generals_pb.common.v1 import common_pb2


class ExperienceStreamingTest:
    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.game_stub = game_pb2_grpc.GameServiceStub(self.channel)
        self.exp_stub = experience_pb2_grpc.ExperienceServiceStub(self.channel)
        self.games_created = []
        self.experiences_received = []
        self.player_tokens = {}  # Store player tokens for each game
        
    def create_test_game(self, collect_experiences: bool = True) -> str:
        """Create a game configured for experience collection."""
        print(f"Creating test game with experience collection={collect_experiences}...")
        
        request = game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(
                width=10,
                height=10,
                max_players=2,
                fog_of_war=False,
                turn_time_ms=100,
                collect_experiences=collect_experiences
            )
        )
        
        try:
            response = self.game_stub.CreateGame(request)
            game_id = response.game_id
            self.games_created.append(game_id)
            print(f"✓ Created game: {game_id}")
            return game_id
        except grpc.RpcError as e:
            print(f"✗ Failed to create game: {e.code()}: {e.details()}")
            raise
            
    def join_game_as_bot(self, game_id: str, player_name: str) -> tuple:
        """Join a game as a bot player."""
        print(f"Joining game {game_id} as {player_name}...")
        
        request = game_pb2.JoinGameRequest(
            game_id=game_id,
            player_name=player_name
        )
        
        try:
            response = self.game_stub.JoinGame(request)
            print(f"✓ Joined as player {response.player_id} with token {response.player_token[:8]}...")
            
            # Store the player token for this game and player
            if game_id not in self.player_tokens:
                self.player_tokens[game_id] = {}
            self.player_tokens[game_id][response.player_id] = response.player_token
            
            return response.player_id, response.player_token
        except grpc.RpcError as e:
            print(f"✗ Failed to join game: {e.code()}: {e.details()}")
            raise
            
    def submit_random_actions(self, game_id: str, num_turns: int = 10) -> None:
        """Submit random actions to generate experiences. Ensures both players submit for each turn."""
        print(f"Submitting actions for {num_turns} turns to game {game_id}...")
        
        # Check if we have tokens for this game
        if game_id not in self.player_tokens or not self.player_tokens[game_id]:
            print(f"✗ No player tokens found for game {game_id}")
            return
            
        # Get the first player's token to fetch game state
        player_id = list(self.player_tokens[game_id].keys())[0]
        player_token = self.player_tokens[game_id][player_id]
        
        # Get game state first to know board size
        state_req = game_pb2.GetGameStateRequest(
            game_id=game_id,
            player_id=player_id,
            player_token=player_token
        )
        
        try:
            state_resp = self.game_stub.GetGameState(state_req)
            board_width = state_resp.state.board.width
            board_height = state_resp.state.board.height
            current_turn = state_resp.state.turn
            
            turns_processed = 0
            
            # Process multiple turns
            for turn in range(num_turns):
                print(f"  Turn {turn + 1}/{num_turns}: submitting for all players...")
                
                # Collect actions for ALL players in this turn
                turn_actions = []
                for player_id, player_token in self.player_tokens[game_id].items():
                    # For simplicity, try to find the general's position from state
                    # In a real implementation, we'd parse the board to find owned tiles
                    # For now, use a simple pattern: move from near spawn points
                    if player_id == 0:
                        # Player 0 typically starts in top-left area
                        from_x = np.random.randint(0, min(3, board_width))
                        from_y = np.random.randint(0, min(3, board_height))
                    else:
                        # Player 1 typically starts in bottom-right area
                        from_x = np.random.randint(max(0, board_width-3), board_width)
                        from_y = np.random.randint(max(0, board_height-3), board_height)
                    
                    # Move to an adjacent tile (4-directional movement)
                    direction = np.random.randint(0, 4)
                    if direction == 0:  # up
                        to_x, to_y = from_x, max(0, from_y - 1)
                    elif direction == 1:  # down
                        to_x, to_y = from_x, min(board_height - 1, from_y + 1)
                    elif direction == 2:  # left
                        to_x, to_y = max(0, from_x - 1), from_y
                    else:  # right
                        to_x, to_y = min(board_width - 1, from_x + 1), from_y
                    
                    # Create action (from is a keyword, so we set it separately)
                    action = game_pb2.Action()
                    action.type = common_pb2.ActionType.ACTION_TYPE_MOVE
                    action.to.x = to_x
                    action.to.y = to_y
                    action.half = False
                    action.turn_number = current_turn + turn  # Set expected turn
                    # Use getattr/setattr for the 'from' field since it's a Python keyword
                    from_coord = getattr(action, 'from')
                    from_coord.x = from_x
                    from_coord.y = from_y
                    
                    turn_actions.append({
                        'player_id': player_id,
                        'player_token': player_token,
                        'action': action
                    })
                
                # Submit all actions for this turn
                all_succeeded = True
                for action_data in turn_actions:
                    action_req = game_pb2.SubmitActionRequest(
                        game_id=game_id,
                        player_id=action_data['player_id'],
                        player_token=action_data['player_token'],
                        action=action_data['action']
                    )
                    
                    try:
                        response = self.game_stub.SubmitAction(action_req)
                        if not response.success:
                            print(f"    Player {action_data['player_id']}: {response.error_message}")
                            all_succeeded = False
                    except grpc.RpcError as e:
                        print(f"    Player {action_data['player_id']} error: {e.code()}")
                        all_succeeded = False
                
                if all_succeeded:
                    turns_processed += 1
                    print(f"    ✓ Turn {turn + 1} processed successfully")
                else:
                    print(f"    ✗ Turn {turn + 1} had errors")
                
                # Small delay between turns to allow processing
                time.sleep(0.1)
                
            print(f"✓ Processed {turns_processed}/{num_turns} turns for game {game_id}")
            
        except grpc.RpcError as e:
            print(f"✗ Failed to get game state: {e.code()}: {e.details()}")
            
    def stream_experiences(self, game_ids: List[str], max_experiences: int = 100) -> int:
        """Stream experiences from the specified games."""
        print(f"\nStreaming experiences from games: {game_ids}")
        print(f"Max experiences to collect: {max_experiences}")
        
        request = experience_pb2.StreamExperiencesRequest(
            game_ids=game_ids,
            follow=True,
            batch_size=32
        )
        
        count = 0
        try:
            # Stream individual experiences
            print("\nTesting StreamExperiences (individual)...")
            stream = self.exp_stub.StreamExperiences(request)
            
            for exp in stream:
                count += 1
                self.experiences_received.append(exp)
                
                # Parse tensor state
                if exp.state and exp.state.data:
                    state_shape = tuple(exp.state.shape)
                    state_data = np.array(exp.state.data).reshape(state_shape)
                    
                    print(f"  Experience {count}: game={exp.game_id}, "
                          f"turn={exp.turn}, player={exp.player_id}, "
                          f"action={exp.action}, reward={exp.reward:.2f}, "
                          f"done={exp.done}, state_shape={state_shape}")
                          
                if count >= max_experiences:
                    print(f"✓ Collected {count} experiences")
                    break
                    
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                print(f"✓ Stream cancelled after {count} experiences")
            else:
                print(f"✗ Stream error: {e.code()}: {e.details()}")
                
        return count
        
    def stream_experience_batches(self, game_ids: List[str], max_batches: int = 10) -> int:
        """Stream experience batches from the specified games."""
        print(f"\nStreaming experience batches from games: {game_ids}")
        print(f"Max batches to collect: {max_batches}")
        
        request = experience_pb2.StreamExperiencesRequest(
            game_ids=game_ids,
            follow=True,
            batch_size=32,
            max_batch_wait_ms=100
        )
        
        batch_count = 0
        exp_count = 0
        
        try:
            print("\nTesting StreamExperienceBatches...")
            stream = self.exp_stub.StreamExperienceBatches(request)
            
            for batch in stream:
                batch_count += 1
                exp_count += len(batch.experiences)
                
                print(f"  Batch {batch_count}: {len(batch.experiences)} experiences, "
                      f"batch_id={batch.batch_id}, stream_id={batch.stream_id}")
                      
                # Check first experience in batch
                if batch.experiences:
                    exp = batch.experiences[0]
                    if exp.state and exp.state.data:
                        state_shape = tuple(exp.state.shape)
                        print(f"    First exp: turn={exp.turn}, player={exp.player_id}, "
                              f"reward={exp.reward:.2f}, state_shape={state_shape}")
                              
                if batch_count >= max_batches:
                    print(f"✓ Collected {batch_count} batches with {exp_count} total experiences")
                    break
                    
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                print(f"✓ Stream cancelled after {batch_count} batches")
            else:
                print(f"✗ Stream error: {e.code()}: {e.details()}")
                
        return exp_count
        
    def get_experience_stats(self, game_ids: List[str]) -> None:
        """Get statistics about collected experiences."""
        print(f"\nGetting experience statistics for games: {game_ids}")
        
        request = experience_pb2.GetExperienceStatsRequest(game_ids=game_ids)
        
        try:
            response = self.exp_stub.GetExperienceStats(request)
            
            print(f"\nExperience Statistics:")
            print(f"  Total experiences: {response.total_experiences}")
            print(f"  Total games: {response.total_games}")
            print(f"  Average reward: {response.average_reward:.4f}")
            print(f"  Min reward: {response.min_reward:.4f}")
            print(f"  Max reward: {response.max_reward:.4f}")
            
            if response.experiences_per_game:
                print(f"\n  Experiences per game:")
                for game_id, count in response.experiences_per_game.items():
                    print(f"    {game_id}: {count}")
                    
            if response.experiences_per_player:
                print(f"\n  Experiences per player:")
                for player_id, count in response.experiences_per_player.items():
                    print(f"    Player {player_id}: {count}")
                    
        except grpc.RpcError as e:
            print(f"✗ Failed to get stats: {e.code()}: {e.details()}")
            
    def run_full_test(self) -> bool:
        """Run a complete end-to-end test."""
        print("=" * 60)
        print("EXPERIENCE STREAMING END-TO-END TEST")
        print("=" * 60)
        
        try:
            # Step 1: Create games with experience collection
            print("\n[Step 1] Creating test games...")
            game1 = self.create_test_game(collect_experiences=True)
            game2 = self.create_test_game(collect_experiences=True)
            
            # Step 2: Add bot players
            print("\n[Step 2] Adding bot players...")
            self.join_game_as_bot(game1, "Bot1")
            self.join_game_as_bot(game1, "Bot2")
            self.join_game_as_bot(game2, "Bot3")
            self.join_game_as_bot(game2, "Bot4")
            
            # Step 3: Submit actions to generate experiences
            print("\n[Step 3] Submitting actions to generate experiences...")
            self.submit_random_actions(game1, num_turns=10)
            self.submit_random_actions(game2, num_turns=10)
            
            # Give time for experience processing
            print("\n[Step 4] Waiting for experience processing...")
            time.sleep(2)
            
            # Step 5: Stream experiences
            print("\n[Step 5] Streaming experiences...")
            exp_count = self.stream_experiences([game1, game2], max_experiences=50)
            
            if exp_count == 0:
                print("✗ No experiences received from individual streaming")
                return False
                
            # Step 6: Stream experience batches
            print("\n[Step 6] Streaming experience batches...")
            batch_exp_count = self.stream_experience_batches([game1, game2], max_batches=5)
            
            if batch_exp_count == 0:
                print("✗ No experiences received from batch streaming")
                return False
                
            # Step 7: Get statistics
            print("\n[Step 7] Getting experience statistics...")
            self.get_experience_stats([game1, game2])
            
            # Summary
            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)
            print(f"✓ Games created: {len(self.games_created)}")
            print(f"✓ Individual experiences streamed: {exp_count}")
            print(f"✓ Batched experiences streamed: {batch_exp_count}")
            print(f"✓ All tests passed!")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.channel.close()


def main():
    # Check if server is running
    print("Checking if game server is running...")
    
    test = ExperienceStreamingTest()
    
    try:
        # Try to connect
        channel = grpc.insecure_channel("localhost:50051")
        grpc.channel_ready_future(channel).result(timeout=2)
        print("✓ Connected to game server")
        channel.close()
    except grpc.FutureTimeoutError:
        print("✗ Could not connect to game server at localhost:50051")
        print("Please start the game server first:")
        print("  go run cmd/game_server/main.go")
        return 1
        
    # Run the test
    success = test.run_full_test()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())