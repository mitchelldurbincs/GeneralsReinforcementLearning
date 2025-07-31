# Python Agent Architecture

This document consolidates recommendations for the Python agent implementation, including both the base architecture design and specific implementation requirements.

## Current State Issues

### BaseAgent Problems
The current `base_agent.py` has several architectural issues:
- **Mixed Responsibilities**: Network management, game lifecycle, state polling, and agent interface combined
- **Poor Error Handling**: Generic exception catching without recovery strategies
- **State Management Issues**: Direct mutation of instance variables, unclear state transitions
- **Import Organization**: Runtime imports to avoid circular dependencies
- **Inefficient Polling**: Tight polling loop with hardcoded intervals

### Missing Components
- Complete random agent implementation
- Multi-agent orchestration
- Move validation in the client
- Game completion handling

## Recommended Architecture

### 1. Separation of Concerns

Create separate classes for different responsibilities:

```python
# connection.py
class GameConnection:
    """Handles gRPC connection lifecycle"""
    def __init__(self, server_address: str)
    def connect(self) -> GameServiceStub
    def disconnect(self)
    def with_retry(self, func, *args, **kwargs)

# game_client.py  
class GameClient:
    """Handles game-specific RPC calls"""
    def __init__(self, connection: GameConnection)
    def create_game(self, config: GameConfig) -> str
    def join_game(self, game_id: str, player_name: str) -> PlayerCredentials
    def get_game_state(self, credentials: PlayerCredentials) -> GameState
    def submit_action(self, credentials: PlayerCredentials, action: Action) -> bool

# game_session.py
class GameSession:
    """Manages a single game session lifecycle"""
    def __init__(self, client: GameClient, credentials: PlayerCredentials)
    def poll_updates(self, callback: Callable[[GameState], None])
    def wait_for_players(self, min_players: int)
    def is_active(self) -> bool

# base_agent.py
class BaseAgent(ABC):
    """Pure abstract agent interface"""
    @abstractmethod
    def select_action(self, game_state: GameState) -> Optional[Action]
    
    @abstractmethod
    def on_game_start(self, initial_state: GameState)
    
    @abstractmethod
    def on_game_end(self, final_state: GameState, winner_id: str)
```

### 2. Improved State Management

```python
from enum import Enum
from dataclasses import dataclass

class AgentState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    WAITING_FOR_GAME = "waiting_for_game"
    IN_GAME = "in_game"
    GAME_ENDED = "game_ended"

@dataclass
class AgentContext:
    state: AgentState
    game_id: Optional[str] = None
    player_id: Optional[str] = None
    player_token: Optional[str] = None
    last_turn: int = -1
    last_game_status: Optional[GameStatus] = None
```

### 3. Better Error Handling

```python
class GameError(Exception):
    """Base exception for game-related errors"""
    pass

class ConnectionError(GameError):
    """Network connection errors"""
    pass

class GameStateError(GameError):
    """Invalid game state errors"""
    pass

class ActionError(GameError):
    """Invalid action errors"""
    pass

def handle_grpc_errors(func):
    """Decorator for consistent gRPC error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise ConnectionError(f"Server unavailable: {e.details()}")
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise ActionError(f"Invalid request: {e.details()}")
            else:
                raise GameError(f"RPC error: {e.code()} - {e.details()}")
    return wrapper
```

## Random Agent Implementation

### Implementation Requirements

The random agent needs to be a complete, autonomous player that can play full games from start to finish.

### File Structure

```
python/
├── generals_agent/
│   ├── __init__.py
│   ├── base_agent.py      # Abstract base class
│   ├── random_agent.py    # Random agent implementation
│   └── game_utils.py      # Validation and utility functions
├── scripts/
│   ├── run_random_match.py    # Launch random vs random games
│   └── tournament_runner.py   # Run multiple matches
└── tests/
    ├── test_agent.py
    └── test_validation.py
```

### Random Agent Implementation

```python
# random_agent.py
import random
from typing import Optional, List
from generals_agent.base_agent import BaseAgent
from generals_agent.game_utils import get_valid_moves

class RandomAgent(BaseAgent):
    def __init__(self, name: str = "RandomAgent"):
        self.name = name
        self.owned_tiles = set()
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def select_action(self, game_state: GameState) -> Optional[Action]:
        """Select a random valid move"""
        valid_moves = get_valid_moves(game_state, self.player_id)
        
        if not valid_moves:
            self.logger.debug("No valid moves available")
            return None
        
        # Choose random move
        move = random.choice(valid_moves)
        
        # Randomly decide between MOVE and MOVE_HALF
        if random.random() < 0.5 and move.source_army > 2:
            move.move_half = True
        
        self.logger.debug(f"Selected move: {move}")
        return move
    
    def on_game_start(self, initial_state: GameState):
        """Initialize agent for new game"""
        self.logger.info(f"Game started as player {self.player_id}")
        self.update_owned_tiles(initial_state)
    
    def on_game_end(self, final_state: GameState, winner_id: str):
        """Handle game end"""
        if winner_id == str(self.player_id):
            self.logger.info("Won the game!")
        else:
            self.logger.info(f"Lost to player {winner_id}")
    
    def update_owned_tiles(self, game_state: GameState):
        """Track tiles owned by this agent"""
        self.owned_tiles.clear()
        for idx, tile in enumerate(game_state.tiles):
            if tile.owner == self.player_id:
                self.owned_tiles.add(idx)
```

### Move Validation Utilities

```python
# game_utils.py
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ValidMove:
    source_idx: int
    target_idx: int
    source_army: int
    move_half: bool = False

def get_owned_tiles(game_state: GameState, player_id: int) -> List[int]:
    """Return list of tile indices owned by player"""
    return [idx for idx, tile in enumerate(game_state.tiles) 
            if tile.owner == player_id]

def get_valid_moves(game_state: GameState, player_id: int) -> List[ValidMove]:
    """Return all possible valid moves for player"""
    valid_moves = []
    board_width = game_state.width
    board_height = game_state.height
    
    for tile_idx in get_owned_tiles(game_state, player_id):
        tile = game_state.tiles[tile_idx]
        if tile.army <= 1:
            continue  # Need at least 2 armies to move
        
        x, y = tile_idx % board_width, tile_idx // board_width
        
        # Check all four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if 0 <= nx < board_width and 0 <= ny < board_height:
                target_idx = ny * board_width + nx
                target_tile = game_state.tiles[target_idx]
                
                # Can't move to mountains
                if target_tile.type == TileType.MOUNTAIN:
                    continue
                
                valid_moves.append(ValidMove(
                    source_idx=tile_idx,
                    target_idx=target_idx,
                    source_army=tile.army
                ))
    
    return valid_moves

def is_valid_move(game_state: GameState, action: Action, player_id: int) -> bool:
    """Validate if a specific action is legal"""
    # Implementation of validation logic
    pass
```

### Multi-Agent Launcher

```python
# run_random_match.py
import asyncio
import argparse
from concurrent.futures import ProcessPoolExecutor
from generals_agent.random_agent import RandomAgent
from generals_agent.agent_runner import AgentRunner

async def run_agent(agent_id: int, game_id: str, server_address: str):
    """Run a single agent in a game"""
    agent = RandomAgent(name=f"RandomAgent_{agent_id}")
    runner = AgentRunner(agent, server_address)
    
    try:
        await runner.join_and_play(game_id)
    except Exception as e:
        print(f"Agent {agent_id} failed: {e}")

async def run_match(server_address: str, num_agents: int = 2):
    """Run a match with multiple random agents"""
    # Create game
    client = GameClient(GameConnection(server_address))
    game_id = await client.create_game(GameConfig(
        width=20,
        height=20,
        fog_of_war=True
    ))
    
    print(f"Created game: {game_id}")
    
    # Launch agents
    tasks = []
    for i in range(num_agents):
        task = asyncio.create_task(run_agent(i, game_id, server_address))
        tasks.append(task)
    
    # Wait for all agents to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Report results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Agent {i} error: {result}")
        else:
            print(f"Agent {i} completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()
    
    for game_num in range(args.games):
        print(f"\nRunning game {game_num + 1}/{args.games}")
        asyncio.run(run_match(args.server, args.agents))
```

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Create `connection.py` with GameConnection class
- [ ] Create `game_client.py` with GameClient class  
- [ ] Create `errors.py` with custom exception hierarchy
- [ ] Create `types.py` with shared type definitions and dataclasses
- [ ] Set up proper package structure to avoid circular imports

### Phase 2: Session Management
- [ ] Create `game_session.py` with GameSession class
- [ ] Implement event-driven architecture with GameEventDispatcher
- [ ] Create configurable polling strategies
- [ ] Add proper state management with AgentContext

### Phase 3: Refactor BaseAgent
- [ ] Extract network code to use new GameConnection/GameClient
- [ ] Simplify BaseAgent to focus only on agent behavior interface
- [ ] Create AgentRunner class that orchestrates agent lifecycle
- [ ] Update imports to use new module structure

### Phase 4: Implement Random Agent
- [ ] Create complete RandomAgent implementation
- [ ] Implement game_utils.py with validation helpers
- [ ] Create run_random_match.py launcher
- [ ] Add tournament runner for multiple games

### Phase 5: Testing and Documentation
- [ ] Write unit tests for each component
- [ ] Create integration tests for full agent lifecycle
- [ ] Add example agents using new architecture
- [ ] Update README with new structure

## Benefits of Refactoring

1. **Testability**: Each component can be tested in isolation
2. **Reusability**: Connection and client code can be shared across different use cases
3. **Maintainability**: Clear separation of concerns makes code easier to understand
4. **Extensibility**: Easy to add new polling strategies, event handlers, or error recovery
5. **Type Safety**: Better type hints and dataclasses reduce runtime errors
6. **Performance**: More efficient state comparison and configurable polling

## Additional Recommendations

### Error Handling and Resilience
- Implement comprehensive error handling for network issues
- Add automatic reconnection with exponential backoff
- Handle server disconnections gracefully
- Log all errors with context for debugging

### Logging and Monitoring
- Add structured logging using Python's logging module
- Include game_id and player_id in all log messages
- Log key events: game start, moves made, game end, errors
- Optional: Send metrics to monitoring system

### Performance Optimization
- Cache game state between turns to reduce parsing
- Batch multiple agent requests when possible
- Use connection pooling for multiple agents
- Profile and optimize hot paths in move selection

### Testing Strategy
- Create unit tests for move validation logic
- Add integration tests for agent-server communication
- Implement replay functionality for debugging
- Create test scenarios for edge cases