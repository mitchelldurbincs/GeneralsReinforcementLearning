# Phase 3 Error System Implementation Plan

## Overview
Phase 3 focuses on systematic error handling improvements in the gRPC server and map generation components. Based on analysis, state.go, visibility.go, and stats.go don't return errors, so they don't need updates.

## Implementation Tasks

### 1. gRPC Server Error Improvements (`internal/grpc/gameserver/server.go`)

#### CreateGame Method
- No major error handling needed (simple ID generation)
- Consider adding validation for game config parameters

#### JoinGame Method
**Current Issues:**
- Line 111: `status.Errorf(codes.NotFound, "game %s not found", req.GameId)`
- Line 128: `status.Error(codes.ResourceExhausted, "game is full")`
- Line 155: `status.Error(codes.Internal, "failed to create game engine")`

**Improvements:**
```go
// Line 111
return nil, status.Errorf(codes.NotFound, "game %s not found: request from player %s", req.GameId, req.PlayerName)

// Line 128
return nil, status.Errorf(codes.ResourceExhausted, "game %s is full: %d/%d players", req.GameId, len(game.players), game.config.MaxPlayers)

// Line 155
return nil, status.Errorf(codes.Internal, "failed to create game engine for game %s: config %dx%d with %d players", req.GameId, game.config.Width, game.config.Height, game.config.MaxPlayers)
```

#### SubmitAction Method
**Current Issues:**
- Multiple error responses use generic messages
- No context about game state when errors occur

**Improvements:**
```go
// Line 195-198 (game not found)
ErrorMessage: fmt.Sprintf("game %s not found", req.GameId),

// Line 205-207 (game not in progress)
ErrorMessage: fmt.Sprintf("game %s is not in progress (status: %s)", req.GameId, game.status),

// Line 222-224 (invalid credentials)
ErrorMessage: fmt.Sprintf("invalid player credentials for game %s: player %d", req.GameId, req.PlayerId),

// Line 235-237 (invalid turn)
ErrorMessage: fmt.Sprintf("invalid turn number for game %s: expected %d, got %d", req.GameId, currentTurn, req.Action.TurnNumber),

// Line 245-247 (action conversion error)
ErrorMessage: fmt.Sprintf("invalid action for game %s player %d: %v", req.GameId, req.PlayerId, err),

// Line 260-262 (validation failed)
ErrorMessage: fmt.Sprintf("action validation failed for game %s player %d turn %d: %v", req.GameId, req.PlayerId, currentTurn, err),

// Line 279-281 (process turn failed)
ErrorMessage: fmt.Sprintf("failed to process turn %d for game %s", currentTurn, req.GameId),
```

#### GetGameState Method
**Current Issues:**
- Line 302: `status.Errorf(codes.NotFound, "game %s not found", req.GameId)`
- Line 314: `status.Error(codes.PermissionDenied, "invalid player credentials")`

**Improvements:**
```go
// Line 302
return nil, status.Errorf(codes.NotFound, "game %s not found: requested by player %d", req.GameId, req.PlayerId)

// Line 314
return nil, status.Errorf(codes.PermissionDenied, "invalid player credentials for game %s: player %d", req.GameId, req.PlayerId)
```

#### Helper Methods
**convertProtoAction:**
- Line 494: Add more context to coordinate validation error
```go
return nil, status.Errorf(codes.InvalidArgument, "move action for player %d requires from and to coordinates", playerID)
```

- Line 507: Add player context to unsupported action type
```go
return nil, status.Errorf(codes.InvalidArgument, "unsupported action type %v for player %d", protoAction.Type, playerID)
```

**processTurn:**
- Line 554: Wrap engine.Step error with context
```go
return fmt.Errorf("game %s turn %d: failed to process %d actions: %w", game.id, game.currentTurn-1, len(actions), err)
```

**startTurnTimer (in timeout handler):**
- Line 609: Add game context to log message (already has some context, could be enhanced)

### 2. Map Generation Error Improvements (`internal/game/mapgen/generator.go`)

#### GenerateMap Method
**Current Issue:**
- Line 62: Good error wrapping but could include map parameters

**Improvement:**
```go
return nil, fmt.Errorf("failed to generate map (%dx%d, %d players): %w", g.config.Width, g.config.Height, g.config.PlayerCount, err)
```

#### placeGenerals Method
**Current Issue:**
- Line 164: Could include more context about map state

**Improvement:**
```go
return nil, fmt.Errorf("failed to place general for player %d on %dx%d map (spacing: %d): %w", pid, g.config.Width, g.config.Height, g.config.MinGeneralSpacing, err)
```

#### findGeneralLocation Method
**Current Issue:**
- Line 244: Could include attempted configuration

**Improvement:**
```go
return GeneralPlacement{}, fmt.Errorf("unable to place general for player %d: no valid locations found on %dx%d map with %d existing generals (min spacing: %d)", len(existing), b.W, b.H, len(existing), g.config.MinGeneralSpacing)
```

## Implementation Priority

1. **High Priority**: gRPC server main methods (JoinGame, SubmitAction, GetGameState)
2. **Medium Priority**: gRPC helper methods and mapgen improvements
3. **Low Priority**: Additional validation and edge cases

## Testing Strategy

1. Unit tests for each modified error path
2. Integration tests to verify error messages in gRPC responses
3. Manual testing with invalid inputs to verify error clarity

## Success Criteria

- All errors include relevant context (game ID, player ID, turn number where applicable)
- Error messages help developers debug issues quickly
- gRPC status codes remain appropriate for each error type
- Error wrapping preserves original error for programmatic checking