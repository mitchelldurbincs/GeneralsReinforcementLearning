# BaseAgent Recommendations and Refactoring Plan

## Overview
The `base_agent.py` file serves as the foundation for all Generals.io agents but has several architectural and code quality issues that make it difficult to maintain and extend. This document provides recommendations for improving the codebase.

## Current Issues

### 1. **Mixed Responsibilities**
The BaseAgent class combines multiple concerns:
- Network connection management
- Game lifecycle management  
- State polling and synchronization
- Abstract agent interface
- Error handling and retry logic

### 2. **Poor Error Handling**
- Generic exception catching without proper recovery strategies
- Inconsistent error propagation
- Missing error context in many places
- No clear distinction between recoverable and fatal errors

### 3. **State Management Issues**
- Direct mutation of instance variables throughout the class
- No clear state transitions or validation
- `_last_status` added as an afterthought with hasattr check
- Multiple state variables that could be out of sync

### 4. **Import Organization**
- Runtime imports to avoid circular dependencies (lines 92, 118, 236)
- sys.path manipulation in module (lines 9-11)
- This suggests poor module structure

### 5. **Polling Architecture**
- Tight polling loop with hardcoded intervals
- No backoff strategy
- Inefficient state comparison logic
- Complex turn/status change detection

### 6. **Code Duplication**
- Game creation logic duplicated in MatchRunner
- Similar logging patterns repeated throughout
- No reusable connection utilities

## Recommended Architecture

### 1. **Separation of Concerns**

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

### 2. **Improved State Management**

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

### 3. **Better Error Handling**

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

### 4. **Event-Driven Architecture**

```python
from typing import Protocol

class GameEventHandler(Protocol):
    def on_state_update(self, old_state: GameState, new_state: GameState): ...
    def on_turn_change(self, turn: int): ...
    def on_game_start(self): ...
    def on_game_end(self, winner_id: str): ...
    def on_error(self, error: Exception): ...

class GameEventDispatcher:
    def __init__(self):
        self.handlers: List[GameEventHandler] = []
    
    def register(self, handler: GameEventHandler):
        self.handlers.append(handler)
    
    def dispatch_state_update(self, old_state: GameState, new_state: GameState):
        for handler in self.handlers:
            try:
                handler.on_state_update(old_state, new_state)
            except Exception as e:
                self._handle_handler_error(handler, e)
```

### 5. **Configurable Polling Strategy**

```python
class PollingStrategy(ABC):
    @abstractmethod
    def get_next_delay(self, attempt: int) -> float: ...

class FixedIntervalPolling(PollingStrategy):
    def __init__(self, interval: float = 0.1):
        self.interval = interval
    
    def get_next_delay(self, attempt: int) -> float:
        return self.interval

class ExponentialBackoffPolling(PollingStrategy):
    def __init__(self, base: float = 0.1, max_delay: float = 5.0):
        self.base = base
        self.max_delay = max_delay
    
    def get_next_delay(self, attempt: int) -> float:
        return min(self.base * (2 ** attempt), self.max_delay)
```

## Implementation Todos

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

### Phase 4: Testing and Migration
- [ ] Write unit tests for each new component
- [ ] Create integration tests for full agent lifecycle
- [ ] Update RandomAgent to use new architecture
- [ ] Update run_random_match.py to use new components
- [ ] Add deprecation warnings to old code paths

### Phase 5: Documentation and Examples
- [ ] Document new architecture with diagrams
- [ ] Create migration guide for existing agents
- [ ] Add example agents using new architecture
- [ ] Update README with new structure

## Benefits of Refactoring

1. **Testability**: Each component can be tested in isolation
2. **Reusability**: Connection and client code can be shared across different use cases
3. **Maintainability**: Clear separation of concerns makes code easier to understand
4. **Extensibility**: Easy to add new polling strategies, event handlers, or error recovery
5. **Type Safety**: Better type hints and dataclasses reduce runtime errors
6. **Performance**: More efficient state comparison and configurable polling

## Migration Strategy

1. Implement new components alongside existing code
2. Add adapter layer to make old interface work with new implementation
3. Gradually migrate agents to use new architecture
4. Once all agents migrated, remove old code
5. This allows incremental migration without breaking existing functionality

## Example Usage with New Architecture

```python
# Using the new architecture
class ImprovedRandomAgent(BaseAgent):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def select_action(self, game_state: GameState) -> Optional[Action]:
        valid_moves = get_valid_moves(game_state, self.player_id)
        return random.choice(valid_moves) if valid_moves else None
    
    def on_game_start(self, initial_state: GameState):
        self.logger.info("Game started!")
    
    def on_game_end(self, final_state: GameState, winner_id: str):
        self.logger.info(f"Game ended. Winner: {winner_id}")

# Running the agent
async def run_agent():
    connection = GameConnection("localhost:50051")
    client = GameClient(connection)
    
    # Create or join game
    game_id = await client.create_game(GameConfig(width=20, height=20))
    credentials = await client.join_game(game_id, "ImprovedRandomAgent")
    
    # Create session and agent
    session = GameSession(client, credentials)
    agent = ImprovedRandomAgent()
    runner = AgentRunner(agent, session)
    
    # Run the agent
    await runner.run()
```

This refactoring will make the codebase more maintainable, testable, and easier to extend for future RL training requirements.