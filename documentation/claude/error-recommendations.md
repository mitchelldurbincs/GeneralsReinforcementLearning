# Error System Overhaul Recommendations

## Overview
This document outlines a comprehensive plan to improve error handling throughout the GeneralsReinforcementLearning codebase by adding proper error context using `fmt.Errorf` with the `%w` verb for error wrapping.

## Phase 1: Foundation (Quick Wins)

- [x] **Update action validation errors** in `internal/game/core/action.go`
  - Wrap all validation errors with contextual information
  - Include player ID, coordinates, and action type
  - Example: `return fmt.Errorf("player %d: move from (%d,%d) to (%d,%d): %w", m.PlayerID, m.FromX, m.FromY, m.ToX, m.ToY, ErrInvalidCoordinates)`

- [x] **Enhance engine error handling** in `internal/game/engine.go`
  - Add game turn context to all error returns
  - Wrap context cancellation errors with game state info
  - Example: `return fmt.Errorf("game turn %d: action processing: %w", e.gs.Turn, err)`

- [x] **Improve movement validation** in `internal/game/core/movement.go`
  - Add movement details to validation errors
  - Include source/destination tile information

## Phase 2: Error Helper Utilities

- [x] **Create error wrapping utilities** in `internal/game/core/errors.go`
  ```go
  // Helper functions for consistent error wrapping
  func WrapActionError(action Action, err error) error
  func WrapGameStateError(turn int, phase string, err error) error
  func WrapPlayerError(playerID int, operation string, err error) error
  ```

- [x] **Add structured error types** for complex scenarios
  ```go
  type GameError struct {
      Turn      int
      PlayerID  int
      Operation string
      Err       error
  }
  ```

## Phase 3: Systematic Updates

- [x] **Update gRPC server errors** in `internal/grpc/gameserver/server.go`
  - Add game ID and request context to all error returns
  - Ensure gRPC status codes include detailed messages

- [x] **Enhance mapgen error context** (already good, but can be improved)
  - Add map dimensions and generation parameters to errors

- [x] **Review and update all error returns** in:
  - `internal/game/state.go` (no errors returned - state management only)
  - `internal/game/visibility.go` (no errors returned - visibility updates only)
  - `internal/game/stats.go` (no errors returned - stats calculations only)

## Phase 4: Logging and Monitoring

- [ ] **Standardize error logging patterns**
  - Always log errors with full context before returning
  - Use structured logging fields consistently
  - Example: `logger.Error().Err(err).Int("player_id", playerID).Str("action", "move").Msg("Action validation failed")`

- [ ] **Add error metrics** for monitoring
  - Track error types and frequencies
  - Identify common failure patterns

## Implementation Priority

**High Priority (Do First):**
- Phase 1: All items (biggest impact on debugging)
- Phase 2: Error wrapping utilities (creates foundation for consistency)

**Medium Priority:**
- Phase 2: Structured error types (for complex error scenarios)
- Phase 3: All systematic improvements

**Low Priority:**
- Phase 4: Logging and monitoring (nice to have for production)

## Example Implementation Pattern

Here's the pattern to follow throughout the codebase:

```go
// Before:
if !b.InBounds(m.FromX, m.FromY) {
    return ErrInvalidCoordinates
}

// After:
if !b.InBounds(m.FromX, m.FromY) {
    return fmt.Errorf("player %d move from (%d,%d): %w", 
        m.PlayerID, m.FromX, m.FromY, ErrInvalidCoordinates)
}
```

This approach maintains sentinel errors for programmatic checks while adding human-readable context for debugging.

## Benefits

1. **Better Debugging**: Full context available in error messages
2. **Error Tracing**: Can trace errors through the call stack with `errors.Is()` and `errors.As()`
3. **Consistency**: Standardized error messages across the codebase
4. **Monitoring**: Easier to identify and track error patterns in production

## Completed Work Summary

### Phase 1 ✅ (Completed in previous commits)
- Updated action validation errors in `internal/game/core/action.go`
- Enhanced engine error handling in `internal/game/engine.go`
- Improved movement validation in `internal/game/core/movement.go`

### Phase 2 ✅ (Completed in previous commits)
- Created error wrapping utilities in `internal/game/core/errors.go`
- Added structured error types for complex scenarios

### Phase 3 ✅ (Completed in this session)
- **gRPC Server**: Enhanced all error messages with game ID, player ID, and turn context
  - JoinGame: Added player name and game capacity details
  - SubmitAction: Added detailed context for each error type
  - GetGameState: Added requester information
  - Helper methods: Added player and action context
- **Map Generation**: Added map dimensions and generation parameters to all errors
- **State/Visibility/Stats**: Confirmed these don't return errors (internal state management only)