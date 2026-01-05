"""
OpenAI Gym environment for Generals.io reinforcement learning.

This environment provides a standard Gym interface for training RL agents
on the Generals game, connecting to the gRPC game server for game management.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import grpc
import logging
from typing import Optional, Tuple, Dict, Any, List
import time

from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2


class GeneralsEnv(gym.Env):
    """
    OpenAI Gym environment for Generals.io game.
    
    The environment interfaces with the gRPC game server to manage games
    and provides observations suitable for neural network processing.
    
    Observation Space:
        Multi-channel 2D grid representing the game board.
        Channels include:
        - 0: Visibility mask (0=fog, 1=visible)
        - 1: Tile ownership (normalized player ID, -1 for neutral)
        - 2: Army count (log-normalized)
        - 3: Tile type - Empty (1 if empty, 0 otherwise)
        - 4: Tile type - Mountain (1 if mountain, 0 otherwise)
        - 5: Tile type - City (1 if city, 0 otherwise)
        - 6: Tile type - General (1 if general, 0 otherwise)
        - 7: Turn counter (normalized)
        - 8: Valid actions mask
    
    Action Space:
        Discrete action space representing all possible moves.
        Action encoding: from_idx * board_size * 5 + to_idx * 5 + move_type
        where move_type is: 0=up, 1=right, 2=down, 3=left, 4=half_move
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        server_address: str = "localhost:50051",
        board_width: int = 15,
        board_height: int = 15,
        max_players: int = 2,
        fog_of_war: bool = True,
        render_mode: Optional[str] = None,
        self_play: bool = False,
        opponent_agent: Optional[Any] = None,
        max_turns: int = 500,
        turn_time_ms: int = 500,
    ):
        """
        Initialize the Generals Gym environment.
        
        Args:
            server_address: Address of the gRPC game server
            board_width: Width of the game board
            board_height: Height of the game board
            max_players: Maximum number of players (2 for now)
            fog_of_war: Whether to enable fog of war
            render_mode: Rendering mode ("human" or "rgb_array")
            self_play: Whether to use self-play (agent vs itself)
            opponent_agent: Opponent agent for training (if not self-play)
            max_turns: Maximum number of turns before episode ends
            turn_time_ms: Time limit per turn in milliseconds
        """
        super().__init__()
        
        self.server_address = server_address
        self.board_width = board_width
        self.board_height = board_height
        self.board_size = board_width * board_height
        self.max_players = max_players
        self.fog_of_war = fog_of_war
        self.render_mode = render_mode
        self.self_play = self_play
        self.opponent_agent = opponent_agent
        self.max_turns = max_turns
        self.turn_time_ms = turn_time_ms
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # gRPC connection
        self.channel = None
        self.stub = None
        self._connect_to_server()
        
        # Game state
        self.game_id = None
        self.player_id = None
        self.player_token = None
        self.opponent_id = None
        self.opponent_token = None
        self.current_state = None
        self.turn_count = 0
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9, board_height, board_width),  # 9 channels
            dtype=np.float32
        )
        
        # Action space: all possible moves from any tile to adjacent tiles
        # Including half moves (5 actions per tile: up, right, down, left, half)
        self.action_space = spaces.Discrete(self.board_size * 5)
        
        # Action mask for invalid actions
        self.valid_actions_mask = None
    
    def _connect_to_server(self):
        """Establish connection to the gRPC game server."""
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = game_pb2_grpc.GameServiceStub(self.channel)
            
            # Test connection with valid config
            test_response = self.stub.CreateGame(game_pb2.CreateGameRequest(
                config=game_pb2.GameConfig(width=5, height=5, max_players=2)
            ))
            # Clean up test game
            # Note: In production, we might want to implement a DeleteGame method
            
            self.logger.info(f"Connected to game server at {self.server_address}")
        except grpc.RpcError as e:
            raise ConnectionError(f"Failed to connect to game server: {e}")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and start a new game.
        
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Clean up previous game if exists
        if self.game_id:
            # TODO: Add cleanup/leave game logic if needed
            pass
        
        # Create new game
        create_response = self.stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(
                width=self.board_width,
                height=self.board_height,
                max_players=self.max_players,
                fog_of_war=self.fog_of_war,
                turn_time_ms=self.turn_time_ms
            )
        ))
        self.game_id = create_response.game_id
        
        # Join as player 1
        join1 = self.stub.JoinGame(game_pb2.JoinGameRequest(
            game_id=self.game_id,
            player_name="RL_Agent"
        ))
        self.player_id = join1.player_id
        self.player_token = join1.player_token
        
        # Join as player 2 (opponent)
        join2 = self.stub.JoinGame(game_pb2.JoinGameRequest(
            game_id=self.game_id,
            player_name="Opponent"
        ))
        self.opponent_id = join2.player_id
        self.opponent_token = join2.player_token
        
        # Wait for game to start
        time.sleep(0.1)
        
        # Get initial state
        state_response = self.stub.GetGameState(game_pb2.GetGameStateRequest(
            game_id=self.game_id,
            player_token=self.player_token
        ))
        self.current_state = state_response.state
        self.turn_count = 0
        
        # Get observation and action mask
        obs = self._get_observation()
        self.valid_actions_mask = self._get_valid_actions_mask()
        
        info = {
            "game_id": self.game_id,
            "player_id": self.player_id,
            "valid_actions_mask": self.valid_actions_mask,
            "turn": self.turn_count
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action index from the action space
            
        Returns:
            observation: New observation after the action
            reward: Reward for the action
            terminated: Whether the episode has ended (win/loss)
            truncated: Whether the episode was cut short (max turns)
            info: Additional information
        """
        # Convert action index to game action
        game_action = self._action_index_to_game_action(action)
        
        if game_action is None:
            # Invalid action, return small negative reward
            obs = self._get_observation()
            return obs, -0.1, False, False, {"invalid_action": True}
        
        # Submit action to game server
        try:
            self.stub.SubmitAction(game_pb2.SubmitActionRequest(
                game_id=self.game_id,
                player_token=self.player_token,
                action=game_action
            ))
        except grpc.RpcError as e:
            self.logger.warning(f"Action submission failed: {e}")
            obs = self._get_observation()
            return obs, -0.1, False, False, {"error": str(e)}
        
        # Submit opponent action (random for now)
        if self.opponent_agent:
            # Use provided opponent agent
            opponent_action = self.opponent_agent.select_action(self.current_state)
            if opponent_action:
                self.stub.SubmitAction(game_pb2.SubmitActionRequest(
                    game_id=self.game_id,
                    player_token=self.opponent_token,
                    action=opponent_action
                ))
        else:
            # Random opponent
            self._submit_random_opponent_action()
        
        # Wait for turn to process
        time.sleep(0.05)
        
        # Get new state
        state_response = self.stub.GetGameState(game_pb2.GetGameStateRequest(
            game_id=self.game_id,
            player_token=self.player_token
        ))
        
        prev_state = self.current_state
        self.current_state = state_response.state
        self.turn_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(prev_state, self.current_state)
        
        # Check if game ended
        terminated = self.current_state.status != common_pb2.GAME_STATUS_IN_PROGRESS
        truncated = self.turn_count >= self.max_turns
        
        # Get new observation
        obs = self._get_observation()
        self.valid_actions_mask = self._get_valid_actions_mask()
        
        info = {
            "turn": self.turn_count,
            "valid_actions_mask": self.valid_actions_mask,
            "game_status": common_pb2.GameStatus.Name(self.current_state.status),
            "winner": self.current_state.winner_id if terminated else None
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Convert game state to observation tensor.
        
        Returns:
            9-channel observation array
        """
        obs = np.zeros((9, self.board_height, self.board_width), dtype=np.float32)
        
        if not self.current_state:
            return obs
        
        board = self.current_state.board
        
        for y in range(self.board_height):
            for x in range(self.board_width):
                idx = y * self.board_width + x
                tile = board.tiles[idx]
                
                # Channel 0: Visibility
                if tile.visible:
                    obs[0, y, x] = 1.0
                
                # Channel 1: Ownership (-1 neutral, 0 us, 1 enemy)
                if tile.owner_id == self.player_id:
                    obs[1, y, x] = 0.5
                elif tile.owner_id >= 0:
                    obs[1, y, x] = 1.0
                else:
                    obs[1, y, x] = 0.0
                
                # Channel 2: Army count (log normalized)
                if tile.army_count > 0:
                    obs[2, y, x] = np.log(tile.army_count + 1) / 10.0
                
                # Channels 3-6: Tile types (one-hot)
                if tile.type == common_pb2.TILE_TYPE_NORMAL:
                    obs[3, y, x] = 1.0
                elif tile.type == common_pb2.TILE_TYPE_MOUNTAIN:
                    obs[4, y, x] = 1.0
                elif tile.type == common_pb2.TILE_TYPE_CITY:
                    obs[5, y, x] = 1.0
                elif tile.type == common_pb2.TILE_TYPE_GENERAL:
                    obs[6, y, x] = 1.0
                
        # Channel 7: Turn counter (normalized)
        obs[7, :, :] = min(self.turn_count / self.max_turns, 1.0)
        
        # Channel 8: Valid actions (computed separately)
        # This will be filled by action masking
        
        return obs
    
    def _get_valid_actions_mask(self) -> np.ndarray:
        """
        Get mask of valid actions for the current state.
        
        Returns:
            Boolean array indicating valid actions
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        if not self.current_state:
            return mask
        
        board = self.current_state.board
        
        # Check all possible moves
        for y in range(self.board_height):
            for x in range(self.board_width):
                idx = y * self.board_width + x
                tile = board.tiles[idx]
                
                # Can only move from tiles we own with armies > 1
                if tile.owner_id != self.player_id or tile.army_count <= 1:
                    continue
                
                # Check all 4 directions + half move
                for direction, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
                    nx, ny = x + dx, y + dy
                    
                    # Check bounds
                    if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                        to_idx = ny * self.board_width + nx
                        to_tile = board.tiles[to_idx]
                        
                        # Can't move to mountains
                        if to_tile.type != common_pb2.TILE_TYPE_MOUNTAIN:
                            # Full move
                            action_idx = idx * 5 + direction
                            mask[action_idx] = True
                            
                            # Half move (always valid if full move is)
                            action_idx = idx * 5 + 4  # 4 = half move flag
                            mask[action_idx] = True
        
        return mask
    
    def _action_index_to_game_action(self, action_idx: int) -> Optional[game_pb2.Action]:
        """
        Convert action index to game action protobuf.
        
        Args:
            action_idx: Index from action space
            
        Returns:
            Game action or None if invalid
        """
        if not self.valid_actions_mask[action_idx]:
            return None
        
        # Decode action
        from_idx = action_idx // 5
        move_info = action_idx % 5
        
        from_x = from_idx % self.board_width
        from_y = from_idx // self.board_width
        
        # Determine move type and direction
        is_half = bool(move_info == 4)
        direction = move_info if move_info < 4 else 0
        
        # Calculate target position
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        if move_info < 4:
            dx, dy = directions[direction]
            to_x = from_x + dx
            to_y = from_y + dy
        else:
            # For half moves, need to find a valid direction
            # This is simplified - in practice we'd track which direction
            for dx, dy in directions:
                to_x = from_x + dx
                to_y = from_y + dy
                if 0 <= to_x < self.board_width and 0 <= to_y < self.board_height:
                    break
        
        # Create action
        action = game_pb2.Action()
        action.type = common_pb2.ACTION_TYPE_MOVE
        action.half = is_half
        
        # Note: 'from' is a reserved keyword in Python, so we use getattr
        from_coord = common_pb2.Coordinate(x=from_x, y=from_y)
        to_coord = common_pb2.Coordinate(x=to_x, y=to_y)
        
        # Use getattr for the 'from' field since it's a reserved keyword
        getattr(action, 'from').CopyFrom(from_coord)
        action.to.CopyFrom(to_coord)
        
        return action
    
    def _submit_random_opponent_action(self):
        """Submit a random valid action for the opponent."""
        # Skip if no opponent token (shouldn't happen but be safe)
        if not self.opponent_token:
            return
            
        # Get opponent's view of the game
        try:
            state_response = self.stub.GetGameState(game_pb2.GetGameStateRequest(
                game_id=self.game_id,
                player_token=self.opponent_token
            ))
            
            # Find a random valid move
            board = state_response.state.board
            valid_moves = []
            
            for y in range(self.board_height):
                for x in range(self.board_width):
                    idx = y * self.board_width + x
                    tile = board.tiles[idx]
                    
                    if tile.owner_id != self.opponent_id or tile.army_count <= 1:
                        continue
                    
                    # Check adjacent tiles
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                            to_idx = ny * self.board_width + nx
                            to_tile = board.tiles[to_idx]
                            
                            if to_tile.type != common_pb2.TILE_TYPE_MOUNTAIN:
                                action = game_pb2.Action()
                                action.type = common_pb2.ACTION_TYPE_MOVE
                                action.half = False
                                
                                from_coord = common_pb2.Coordinate(x=x, y=y)
                                to_coord = common_pb2.Coordinate(x=nx, y=ny)
                                getattr(action, 'from').CopyFrom(from_coord)
                                action.to.CopyFrom(to_coord)
                                
                                valid_moves.append(action)
            
            if valid_moves:
                import random
                selected_action = random.choice(valid_moves)
                self.stub.SubmitAction(game_pb2.SubmitActionRequest(
                    game_id=self.game_id,
                    player_token=self.opponent_token,
                    action=selected_action
                ))
        except grpc.RpcError as e:
            # Log but don't fail - opponent might not be able to move
            self.logger.debug(f"Opponent action failed: {e.code()}: {e.details()}")
    
    def _calculate_reward(self, prev_state: game_pb2.GameState, curr_state: game_pb2.GameState) -> float:
        """
        Calculate reward based on state transition.
        
        Reward structure:
        - Win: +100
        - Loss: -100
        - Capture territory: +1 per tile
        - Lose territory: -1 per tile
        - Increase army: +0.01 per army
        - Decrease army: -0.01 per army
        - Capture city: +5
        - Lose city: -5
        - Eliminate opponent: +50
        
        Args:
            prev_state: Previous game state
            curr_state: Current game state
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Check for game end
        if curr_state.status != common_pb2.GAME_STATUS_IN_PROGRESS:
            if curr_state.winner_id == self.player_id:
                return 100.0  # Win
            else:
                return -100.0  # Loss
        
        # Find our player stats
        prev_stats = None
        curr_stats = None
        
        for player in prev_state.players:
            if player.id == self.player_id:
                prev_stats = player
                break
        
        for player in curr_state.players:
            if player.id == self.player_id:
                curr_stats = player
                break
        
        if prev_stats and curr_stats:
            # Territory change
            tile_diff = curr_stats.tile_count - prev_stats.tile_count
            reward += tile_diff * 1.0
            
            # Army change
            army_diff = curr_stats.army_count - prev_stats.army_count
            reward += army_diff * 0.01
            
            # Check if opponent was eliminated
            for player in prev_state.players:
                if player.id != self.player_id and player.status == common_pb2.PLAYER_STATUS_ACTIVE:
                    # Check if this player is now eliminated
                    for curr_player in curr_state.players:
                        if curr_player.id == player.id and curr_player.status != common_pb2.PLAYER_STATUS_ACTIVE:
                            reward += 50.0  # Eliminated an opponent
        
        return reward
    
    def render(self):
        """Render the game state (optional, for debugging)."""
        if self.render_mode == "human" and self.current_state:
            self._print_board()
    
    def _print_board(self):
        """Print the board in text format for debugging."""
        if not self.current_state:
            return
        
        board = self.current_state.board
        print(f"\nTurn {self.turn_count}")
        print("=" * (self.board_width * 4 + 1))
        
        for y in range(self.board_height):
            row = "|"
            for x in range(self.board_width):
                idx = y * self.board_width + x
                tile = board.tiles[idx]
                
                if not tile.visible:
                    cell = " ? "
                elif tile.type == common_pb2.TILE_TYPE_MOUNTAIN:
                    cell = "###"
                elif tile.owner_id == self.player_id:
                    cell = f" {tile.army_count:2}"
                elif tile.owner_id >= 0:
                    cell = f"-{tile.army_count:2}"
                else:
                    cell = " . "
                
                row += cell + "|"
            print(row)
        
        print("=" * (self.board_width * 4 + 1))
    
    def close(self):
        """Clean up resources."""
        if self.channel:
            self.channel.close()


# Register the environment with Gym
gym.register(
    id='Generals-v0',
    entry_point='generals_gym.generals_env:GeneralsEnv',
    max_episode_steps=500,
)