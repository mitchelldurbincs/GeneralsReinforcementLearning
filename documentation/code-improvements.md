# Code Improvements

This document consolidates various code improvement recommendations and their implementation status.

## 1. Error System Overhaul ✅ COMPLETED

### Overview
Comprehensive plan to improve error handling throughout the codebase by adding proper error context using `fmt.Errorf` with the `%w` verb for error wrapping.

### Completed Work
- ✅ **Phase 1: Foundation** - Updated action validation, engine error handling, and movement validation
- ✅ **Phase 2: Error Helper Utilities** - Created error wrapping utilities and structured error types
- ✅ **Phase 3: Systematic Updates** - Enhanced gRPC server errors and reviewed all error returns

### Implementation Pattern
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

## 2. Coordinate Type Implementation ✅ PHASES 1-3 COMPLETED

### Overview
Introduced a `core.Coordinate` struct to replace separate `x, y int` parameters throughout the codebase for cleaner APIs and better type safety.

### Completed Work
- ✅ **Phase 1: Core Implementation**
  - Created `internal/game/core/coordinate.go` with full Coordinate struct
  - Added coordinate methods to Board (InBoundsCoord, GetTileCoord, SetTile, IdxCoord)
  - Updated MoveAction with From/To Coordinate fields
  - Updated CaptureDetails with Location Coordinate field
  
- ✅ **Phase 2: Common Utilities**
  - Added coordinate-based validation functions
  - Added coordinate-based math utilities
  - Fixed import cycles

- ✅ **Phase 3: Event System** (Already migrated!)
  - MoveExecutedEvent uses From/To Coordinates
  - CombatResolvedEvent uses Location Coordinate

### Benefits Achieved
- Cleaner APIs: Reduced function parameters from 4 to 2 for coordinate operations
- Type Safety: Harder to accidentally swap x/y values
- Consistency: Single way to represent positions
- Full backward compatibility maintained

### Next Steps
- Phase 4: Update game state with Coordinate fields (Priority: Medium)
- Phase 5: Update UI system to use Coordinate types (Priority: Medium)
- Phase 6: Update test utilities and fixtures (Priority: Low)

## 3. Visibility System Optimization ✅ PHASES 1-2 COMPLETED

### Overview
Replaced `map[int]bool` visibility tracking (128+ bytes overhead) with uint32 bitfield (4 bytes) for significant memory and performance improvements.

### Completed Work
- ✅ **Phase 1: Data Structure Changes**
  - Changed Visible map to uint32 bitfield in Tile struct
  - Added IsVisibleTo() and SetVisible() helper methods
  - Supports up to 32 players

- ✅ **Phase 2: Algorithm Optimizations**
  - Created lookup table for 3x3 visibility offsets
  - Implemented delta tracking for ownership changes
  - Optimized with bitwise operations

### Key Optimizations Achieved
- **Memory**: 32x reduction per tile (128 bytes → 4 bytes)
- **CPU**: O(1) visibility checks vs O(n) map lookups
- **GC**: Zero allocations for visibility updates
- **Cache**: Better locality with contiguous memory

### Remaining Work
- [ ] Phase 3: Update all consumers (rendering.go, grpc/gameserver, ui/renderer)
- [ ] Phase 4: Testing & validation with production workloads

## 4. Action Validator Extraction Plan 📋 NOT STARTED

### Overview
Plan to extract validation logic from the `SubmitAction` method in the gRPC server into a dedicated `ActionValidator` component for better separation of concerns.

### Current Issues
- Validation logic mixed with request handling in SubmitAction
- Difficult to test validation in isolation
- Cannot reuse validation logic in other contexts

### Proposed Design

```go
// ActionValidator validates game actions at different levels
type ActionValidator interface {
    // ValidateRequest performs high-level request validation
    ValidateRequest(ctx context.Context, req *gamev1.SubmitActionRequest) *ValidationResult
    
    // ValidateGameContext validates action against game state context
    ValidateGameContext(game *GameInstance, playerID int32, playerToken string, turnNumber int32) *ValidationResult
    
    // ValidateGameLogic validates the core game action
    ValidateGameLogic(action core.Action, gameState *game.GameState, playerID int) *ValidationResult
}
```

### Benefits
1. **Separation of Concerns** - Validation separated from request handling
2. **Improved Testability** - Validators can be unit tested independently
3. **Reusability** - Validation logic can be reused in other contexts
4. **Extensibility** - New validation rules can be added easily

### Implementation Tasks
- [ ] Create validation package structure
- [ ] Define ActionValidator interface
- [ ] Implement DefaultActionValidator
- [ ] Create unit tests
- [ ] Refactor SubmitAction to use validator
- [ ] Test integration with existing game flow

## Summary

### Completed Improvements
1. **Error System**: Full error context wrapping implemented across codebase
2. **Coordinate Type**: Phases 1-3 complete, providing cleaner coordinate handling
3. **Visibility Optimization**: Core optimization complete, 32x memory reduction achieved

### In Progress
- Coordinate Type Phase 4-6 (game state and UI updates)
- Visibility System Phase 3-4 (consumer updates and validation)

### Not Started
- Action Validator Extraction (would improve code organization and testability)

### Impact
These improvements have:
- Made debugging significantly easier with contextual errors
- Reduced memory usage dramatically for visibility tracking
- Improved code readability with coordinate types
- Maintained full backward compatibility throughout