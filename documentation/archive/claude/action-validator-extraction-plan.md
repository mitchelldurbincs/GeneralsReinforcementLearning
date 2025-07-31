# Action Validator Extraction Plan

## Overview

This document outlines the plan to extract validation logic from the `SubmitAction` method in the gRPC server into a dedicated `ActionValidator` component. The current implementation mixes multiple validation concerns within the server method, making it difficult to test, reuse, and maintain. By extracting this logic into a separate component, we will improve code organization, testability, and enable validation reuse across different parts of the system.

## Current State Analysis

### Validation Logic in SubmitAction

The `SubmitAction` method in `/internal/grpc/gameserver/server.go` currently performs the following validations:

1. **Game Existence Validation** (lines 196-204)
   - Verifies the game ID exists in the game manager
   - Returns `ERROR_CODE_GAME_NOT_FOUND` if not found

2. **Idempotency Check** (lines 206-215)
   - Checks if the request has been processed before using idempotency key
   - Returns cached response if found

3. **Game Phase Validation** (lines 217-234)
   - Ensures game is in `GAME_PHASE_RUNNING` phase
   - Returns `ERROR_CODE_INVALID_PHASE` or `ERROR_CODE_GAME_OVER` based on phase

4. **Player Authentication** (lines 236-254)
   - Verifies player ID and token match
   - Returns `ERROR_CODE_INVALID_PLAYER` if authentication fails

5. **Turn Number Validation** (lines 256-272)
   - Checks if the action is for the current turn
   - Returns `ERROR_CODE_INVALID_TURN` if turn mismatch

6. **Action Conversion** (lines 274-286)
   - Converts protobuf action to core action
   - Handles conversion errors

7. **Game Logic Validation** (lines 288-306)
   - Calls `action.Validate()` with game board and player ID
   - Returns validation errors from the core action

### Core Action Validation

The `MoveAction.Validate()` method in `/internal/game/core/action.go` performs:

1. **Coordinate Bounds Checking** (lines 57-64)
   - Verifies from/to coordinates are within board bounds
   - Returns `ErrInvalidCoordinates` if out of bounds

2. **Self-Move Check** (lines 66-69)
   - Ensures from and to tiles are different
   - Returns `ErrMoveToSelf` if same tile

3. **Adjacency Validation** (lines 71-76)
   - Checks if tiles are orthogonally adjacent
   - Returns `ErrNotAdjacent` if not adjacent

4. **Ownership Validation** (lines 78-84)
   - Verifies player owns the source tile
   - Returns `ErrNotOwned` if not owned

5. **Army Count Validation** (lines 86-89)
   - Ensures source tile has sufficient armies (>1)
   - Returns `ErrInsufficientArmy` if insufficient

6. **Terrain Validation** (lines 91-98)
   - Checks if target tile is a mountain
   - Returns `ErrTargetIsMountain` if mountain

## Design for ActionValidator

### Interface Definition

```go
// Package validation provides action validation for the game server
package validation

import (
    "context"
    
    "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
    "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
    gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// ActionValidator validates game actions at different levels
type ActionValidator interface {
    // ValidateRequest performs high-level request validation
    // Returns a ValidationResult with any errors and suggested response
    ValidateRequest(ctx context.Context, req *gamev1.SubmitActionRequest) *ValidationResult
    
    // ValidateGameContext validates action against game state context
    // This includes phase, turn, and player authentication
    ValidateGameContext(game *GameInstance, playerID int32, playerToken string, turnNumber int32) *ValidationResult
    
    // ValidateGameLogic validates the core game action
    // This delegates to the action's own Validate method
    ValidateGameLogic(action core.Action, gameState *game.GameState, playerID int) *ValidationResult
}

// ValidationResult contains the outcome of validation
type ValidationResult struct {
    Valid        bool
    ErrorCode    commonv1.ErrorCode
    ErrorMessage string
    CachedResponse *gamev1.SubmitActionResponse // For idempotency
}

// GameInstance represents the minimal game data needed for validation
type GameInstance interface {
    GetID() string
    GetPhase() commonv1.GamePhase
    GetCurrentTurn() int32
    GetPlayers() []PlayerInfo
    GetEngine() *game.Engine
    CheckIdempotency(key string) *gamev1.SubmitActionResponse
}

// PlayerInfo represents player authentication data
type PlayerInfo interface {
    GetID() int32
    GetToken() string
    GetName() string
}
```

### Concrete Implementation

```go
// DefaultActionValidator provides standard validation implementation
type DefaultActionValidator struct {
    gameManager GameManager
    logger      zerolog.Logger
}

// NewDefaultActionValidator creates a new validator instance
func NewDefaultActionValidator(gameManager GameManager, logger zerolog.Logger) *DefaultActionValidator {
    return &DefaultActionValidator{
        gameManager: gameManager,
        logger:      logger,
    }
}

// ValidateRequest performs initial request validation
func (v *DefaultActionValidator) ValidateRequest(ctx context.Context, req *gamev1.SubmitActionRequest) *ValidationResult {
    // Validate game exists
    game, exists := v.gameManager.GetGame(req.GameId)
    if !exists {
        return &ValidationResult{
            Valid:        false,
            ErrorCode:    commonv1.ErrorCode_ERROR_CODE_GAME_NOT_FOUND,
            ErrorMessage: fmt.Sprintf("game %s not found", req.GameId),
        }
    }
    
    // Check idempotency
    if req.IdempotencyKey != "" {
        if cached := game.CheckIdempotency(req.IdempotencyKey); cached != nil {
            v.logger.Debug().
                Str("game_id", req.GameId).
                Str("idempotency_key", req.IdempotencyKey).
                Msg("Returning cached response for idempotent request")
            return &ValidationResult{
                Valid:          true,
                CachedResponse: cached,
            }
        }
    }
    
    return &ValidationResult{Valid: true}
}

// ValidateGameContext validates against game state
func (v *DefaultActionValidator) ValidateGameContext(game GameInstance, playerID int32, playerToken string, turnNumber int32) *ValidationResult {
    // Check game phase
    currentPhase := game.GetPhase()
    if currentPhase != commonv1.GamePhase_GAME_PHASE_RUNNING {
        errorCode := commonv1.ErrorCode_ERROR_CODE_INVALID_PHASE
        if currentPhase == commonv1.GamePhase_GAME_PHASE_ENDED {
            errorCode = commonv1.ErrorCode_ERROR_CODE_GAME_OVER
        }
        
        return &ValidationResult{
            Valid:        false,
            ErrorCode:    errorCode,
            ErrorMessage: fmt.Sprintf("game %s cannot accept actions in %s phase", game.GetID(), currentPhase.String()),
        }
    }
    
    // Authenticate player
    authenticated := false
    for _, p := range game.GetPlayers() {
        if p.GetID() == playerID && p.GetToken() == playerToken {
            authenticated = true
            break
        }
    }
    
    if !authenticated {
        return &ValidationResult{
            Valid:        false,
            ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_PLAYER,
            ErrorMessage: fmt.Sprintf("invalid player credentials for game %s: player %d", game.GetID(), playerID),
        }
    }
    
    // Validate turn number
    if turnNumber != game.GetCurrentTurn() {
        return &ValidationResult{
            Valid:        false,
            ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
            ErrorMessage: fmt.Sprintf("invalid turn number for game %s: expected %d, got %d", 
                game.GetID(), game.GetCurrentTurn(), turnNumber),
        }
    }
    
    return &ValidationResult{Valid: true}
}

// ValidateGameLogic validates the action against game rules
func (v *DefaultActionValidator) ValidateGameLogic(action core.Action, gameState *game.GameState, playerID int) *ValidationResult {
    if action == nil {
        // Nil action is valid (represents no action/wait)
        return &ValidationResult{Valid: true}
    }
    
    // Validate using the action's built-in validation
    if err := action.Validate(gameState.Board, playerID); err != nil {
        return &ValidationResult{
            Valid:        false,
            ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
            ErrorMessage: fmt.Sprintf("action validation failed: %v", err),
        }
    }
    
    return &ValidationResult{Valid: true}
}
```

### Validation Chain Pattern (Optional Enhancement)

```go
// ValidationChain allows composing multiple validators
type ValidationChain struct {
    validators []ValidationStep
}

// ValidationStep represents a single validation step
type ValidationStep interface {
    Validate(ctx context.Context, data ValidationData) *ValidationResult
    Name() string
}

// ValidationData carries all data needed for validation
type ValidationData struct {
    Request     *gamev1.SubmitActionRequest
    Game        GameInstance
    Action      core.Action
    GameState   *game.GameState
}

// AddStep adds a validation step to the chain
func (c *ValidationChain) AddStep(step ValidationStep) {
    c.validators = append(c.validators, step)
}

// Validate runs all validation steps in order
func (c *ValidationChain) Validate(ctx context.Context, data ValidationData) *ValidationResult {
    for _, validator := range c.validators {
        result := validator.Validate(ctx, data)
        if !result.Valid {
            return result
        }
    }
    return &ValidationResult{Valid: true}
}
```

## Integration Approach

### Modified SubmitAction Method

```go
// SubmitAction with extracted validation logic
func (s *Server) SubmitAction(ctx context.Context, req *gamev1.SubmitActionRequest) (*gamev1.SubmitActionResponse, error) {
    log.Debug().
        Str("game_id", req.GameId).
        Int32("player_id", req.PlayerId).
        Str("idempotency_key", req.IdempotencyKey).
        Msg("Received action submission")
    
    // Step 1: Validate request
    result := s.validator.ValidateRequest(ctx, req)
    if !result.Valid {
        resp := &gamev1.SubmitActionResponse{
            Success:      false,
            ErrorCode:    result.ErrorCode,
            ErrorMessage: result.ErrorMessage,
        }
        s.storeIdempotency(req, resp)
        return resp, nil
    }
    
    // Handle cached response
    if result.CachedResponse != nil {
        return result.CachedResponse, nil
    }
    
    // Get game instance
    game, _ := s.gameManager.GetGame(req.GameId)
    
    // Step 2: Validate game context
    result = s.validator.ValidateGameContext(game, req.PlayerId, req.PlayerToken, req.Action.TurnNumber)
    if !result.Valid {
        resp := &gamev1.SubmitActionResponse{
            Success:      false,
            ErrorCode:    result.ErrorCode,
            ErrorMessage: result.ErrorMessage,
        }
        s.storeIdempotency(req, resp)
        return resp, nil
    }
    
    // Step 3: Convert action
    coreAction, err := convertProtoAction(req.Action, req.PlayerId)
    if err != nil {
        resp := &gamev1.SubmitActionResponse{
            Success:      false,
            ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
            ErrorMessage: fmt.Sprintf("invalid action for game %s player %d: %v", req.GameId, req.PlayerId, err),
        }
        s.storeIdempotency(req, resp)
        return resp, nil
    }
    
    // Step 4: Validate game logic
    game.mu.Lock()
    engineState := game.engine.GameState()
    game.mu.Unlock()
    
    result = s.validator.ValidateGameLogic(coreAction, &engineState, int(req.PlayerId))
    if !result.Valid {
        resp := &gamev1.SubmitActionResponse{
            Success:      false,
            ErrorCode:    result.ErrorCode,
            ErrorMessage: result.ErrorMessage,
        }
        s.storeIdempotency(req, resp)
        return resp, nil
    }
    
    // Process the action (existing logic)
    allSubmitted := game.collectAction(req.PlayerId, coreAction)
    
    if allSubmitted {
        if err := game.processTurn(ctx); err != nil {
            log.Error().Err(err).
                Str("game_id", req.GameId).
                Int32("turn", game.currentTurn).
                Msg("Failed to process turn")
            
            resp := &gamev1.SubmitActionResponse{
                Success:      false,
                ErrorCode:    commonv1.ErrorCode_ERROR_CODE_UNSPECIFIED,
                ErrorMessage: fmt.Sprintf("failed to process turn %d for game %s", game.currentTurn, req.GameId),
            }
            s.storeIdempotency(req, resp)
            return resp, nil
        }
        
        game.broadcastUpdates(s)
        
        if game.config.TurnTimeMs > 0 && game.CurrentPhase() == commonv1.GamePhase_GAME_PHASE_RUNNING {
            game.startTurnTimer(ctx, time.Duration(game.config.TurnTimeMs)*time.Millisecond)
        }
    }
    
    resp := &gamev1.SubmitActionResponse{
        Success:        true,
        NextTurnNumber: game.currentTurn + 1,
    }
    s.storeIdempotency(req, resp)
    return resp, nil
}
```

### Server Initialization Update

```go
// NewServer creates a new game server with validator
func NewServer() *Server {
    gameManager := NewGameManager()
    validator := validation.NewDefaultActionValidator(gameManager, log.Logger)
    
    return &Server{
        gameManager: gameManager,
        validator:   validator,
    }
}
```

## Benefits

1. **Separation of Concerns**
   - Validation logic is separated from request handling
   - Each validation type has its own method
   - Easy to understand and modify

2. **Improved Testability**
   - Validators can be unit tested independently
   - Mock implementations can be created for testing
   - Different validation scenarios can be tested in isolation

3. **Reusability**
   - Validation logic can be reused in other contexts (e.g., AI agents, batch processing)
   - Common validation patterns can be extracted and shared

4. **Extensibility**
   - New validation rules can be added without modifying the server
   - Custom validators can be implemented for different game modes
   - Validation chain pattern allows flexible composition

5. **Better Error Handling**
   - Consistent error responses across all validation types
   - Clear error codes and messages
   - Easier to track validation failures

6. **Performance**
   - Early validation exit reduces unnecessary processing
   - Validation can be optimized independently
   - Potential for parallel validation in the future

## Considerations

1. **Interface Design**
   - The interfaces should be minimal but complete
   - Avoid exposing internal implementation details
   - Consider future validation needs

2. **Error Handling**
   - Maintain backward compatibility with existing error codes
   - Ensure error messages are helpful for debugging
   - Log validation failures appropriately

3. **Testing Strategy**
   - Create comprehensive unit tests for each validator
   - Test edge cases and error conditions
   - Mock dependencies appropriately

4. **Migration Path**
   - Implement validators alongside existing code first
   - Gradually migrate validation logic
   - Ensure no regression in functionality

5. **Performance Impact**
   - Additional interface calls may have minor overhead
   - Benefits of cleaner code outweigh minimal performance cost
   - Can optimize hot paths if needed

## Task List

- [ ] Create validation package structure (`internal/grpc/validation/`)
- [ ] Define ActionValidator interface and ValidationResult types
- [ ] Implement GameInstance and PlayerInfo interfaces for existing types
- [ ] Create DefaultActionValidator with request validation logic
- [ ] Implement game context validation method
- [ ] Implement game logic validation method
- [ ] Create comprehensive unit tests for DefaultActionValidator
- [ ] Add mock implementations for testing
- [ ] Update Server struct to include validator field
- [ ] Modify NewServer to initialize validator
- [ ] Refactor SubmitAction to use validator
- [ ] Extract idempotency storage helper method
- [ ] Test integration with existing game flow
- [ ] Update any documentation affected by changes
- [ ] Consider implementing ValidationChain pattern (optional enhancement)
- [ ] Add performance benchmarks to ensure no regression
- [ ] Create integration tests for full validation flow

## Future Enhancements

1. **Async Validation**
   - Some validations could be performed in parallel
   - Useful for expensive validations (e.g., anti-cheat checks)

2. **Validation Metrics**
   - Track validation failure rates
   - Monitor performance of validation steps
   - Alert on unusual validation patterns

3. **Dynamic Validation Rules**
   - Load validation rules from configuration
   - Support different rules for different game modes
   - A/B test validation strategies

4. **Client-Side Validation**
   - Share validation logic with clients
   - Reduce server load with early validation
   - Improve user experience with instant feedback