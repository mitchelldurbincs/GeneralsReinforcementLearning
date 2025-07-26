# Random Agent Implementation Recommendations

## Overview
This document outlines the requirements and implementation steps for creating random agents that can play against each other on the gRPC server.

## Current State

### ✅ Working Components
- **gRPC Server** (`cmd/grpc_server/`): Fully functional with support for multiple concurrent games, turn-based processing, and streaming updates
- **Python Client Infrastructure** (`python/`): Basic gRPC connection, protobuf handling, and move submission capabilities
- **Game Engine**: Complete game logic with fog of war, turn processing, and win conditions

### ❌ Missing Components
- Complete random agent implementation
- Multi-agent orchestration
- Move validation in the client
- Game completion handling

## Implementation Requirements

### 1. Random Agent Implementation

The random agent needs to be a complete, autonomous player that can play full games from start to finish.

#### Specific TODOs:
- [ ] Create `python/generals_agent/base_agent.py` with abstract `BaseAgent` class
  - Define abstract methods: `select_action()`, `on_game_start()`, `on_game_end()`
  - Handle connection management and authentication
  - Implement game state tracking

- [ ] Create `python/generals_agent/random_agent.py` implementing `BaseAgent`
  - Implement `select_action()` to choose random valid moves
  - Track owned tiles and their army counts
  - Filter moves to only valid actions (owned tiles with >1 army)
  - Handle both MOVE and MOVE_HALF action types

- [ ] Implement game loop in the agent
  - Connect to server and join game
  - Main loop: get state → select action → submit action → wait for next turn
  - Handle game end conditions (win/loss/disconnect)
  - Implement exponential backoff for retries

- [ ] Add turn synchronization
  - Wait for turn notification via streaming or polling
  - Handle turn timeouts gracefully
  - Ensure actions are submitted within time limits

### 2. Multi-Agent Launcher

Create a system to run multiple agents simultaneously for training and testing.

#### Specific TODOs:
- [ ] Create `python/scripts/run_random_match.py`
  - Parse command line arguments (server address, number of games, etc.)
  - Create and configure game on server
  - Launch two RandomAgent instances as separate processes/threads
  - Monitor game progress and collect results

- [ ] Implement agent process management
  - Use multiprocessing or threading for concurrent agents
  - Handle agent crashes and restarts
  - Implement clean shutdown on game completion

- [ ] Add game configuration options
  - Board size selection
  - Fog of war toggle
  - Turn time limits
  - Number of games to play

- [ ] Implement match statistics collection
  - Track win rates
  - Average game length
  - Move counts per game
  - Time per turn

### 3. Move Validation Logic

Ensure agents only attempt legal moves to reduce server load and improve efficiency.

#### Specific TODOs:
- [ ] Create `python/generals_agent/game_utils.py` for validation helpers
  - `get_owned_tiles(game_state, player_id)`: Return list of owned coordinates
  - `get_valid_moves(game_state, player_id)`: Return all possible valid actions
  - `is_valid_move(game_state, action, player_id)`: Validate specific action

- [ ] Implement movement validation
  - Check source tile is owned by player
  - Verify source tile has >1 army (or >2 for half moves)
  - Ensure destination is adjacent (Manhattan distance = 1)
  - Check destination is not a mountain

- [ ] Add fog of war considerations
  - Only validate moves based on visible information
  - Handle moves into fog appropriately
  - Update internal state based on move results

- [ ] Implement action prioritization (optional enhancement)
  - Prefer moves that capture enemy territory
  - Consider moves toward enemy general if visible
  - Avoid moves that leave territory undefended

## Additional Recommendations

### Error Handling and Resilience
- [ ] Implement comprehensive error handling for network issues
- [ ] Add automatic reconnection with exponential backoff
- [ ] Handle server disconnections gracefully
- [ ] Log all errors with context for debugging

### Logging and Monitoring
- [ ] Add structured logging using Python's logging module
- [ ] Include game_id and player_id in all log messages
- [ ] Log key events: game start, moves made, game end, errors
- [ ] Optional: Send metrics to monitoring system

### Performance Optimization
- [ ] Cache game state between turns to reduce parsing
- [ ] Batch multiple agent requests when possible
- [ ] Use connection pooling for multiple agents
- [ ] Profile and optimize hot paths in move selection

### Testing
- [ ] Create unit tests for move validation logic
- [ ] Add integration tests for agent-server communication
- [ ] Implement replay functionality for debugging
- [ ] Create test scenarios for edge cases

## Example Implementation Structure

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

## Next Steps

1. Start with implementing the `BaseAgent` class to establish the interface
2. Create the `RandomAgent` with basic move selection
3. Build the launcher script for two-agent games
4. Iteratively add validation, error handling, and monitoring
5. Test with multiple concurrent games to ensure stability