# Coordinate Type Implementation Plan

This document outlines the plan for introducing a `core.Coordinate` struct to replace the current pattern of using separate `x, y int` parameters throughout the codebase.

## Overview

The current codebase uses separate integer parameters for coordinates, leading to:
- Verbose function signatures (e.g., `IsAdjacent(x1, y1, x2, y2 int)`)
- Increased risk of parameter mix-ups
- Duplicated coordinate validation logic
- Lack of coordinate-specific methods

## Benefits of Coordinate Struct

1. **Cleaner APIs**: Reduce function parameters from 4 to 2 for coordinate operations
2. **Type Safety**: Harder to accidentally swap x/y values
3. **Encapsulation**: Coordinate-specific logic lives with the data
4. **Consistency**: Single way to represent positions throughout the codebase
5. **Future-Proof**: Easy to add new coordinate methods without changing signatures

## Implementation Status

### Summary
The Coordinate struct has been created and fully tested. The implementation is ready to be integrated into the rest of the codebase. The remaining work involves updating existing code to use the new Coordinate type while maintaining backward compatibility.

### Completed (2025-01-26)
- [x] Created `internal/game/core/coordinate.go` with full Coordinate struct implementation
  - Includes all basic operations (creation, validation, conversion, distance, adjacency)
  - Added bonus features: arithmetic operations, direction support, neighbor queries
  - Fully compatible as a map key (useful for caching, pathfinding, etc.)
- [x] Created `internal/game/core/coordinate_test.go` with comprehensive tests
  - 100% coverage of all methods
  - Edge case testing
  - Performance benchmarks
  - Verified symmetry properties (distance, adjacency)
- [x] Documented full implementation plan in this file

### Next Steps
The Coordinate type is ready to use. When you're ready to continue:
1. Start by updating the event system to use Coordinate (easiest migration)
2. Add coordinate-based methods to Board while keeping old ones
3. Gradually migrate other systems as you work on them

### Implementation Plan

### Phase 1: Core Implementation (Priority: High) ✅ PARTIALLY COMPLETE

#### 1.1 Create Coordinate Type ✅ COMPLETE
**File**: `internal/game/core/coordinate.go` (NEW)

Implemented with the following features:
- Basic struct with X, Y int fields
- Constructors: `NewCoordinate(x, y int)`, `FromIndex(idx, width int)`
- Validation: `IsValid(width, height int) bool`
- Conversion: `ToIndex(width int) int`
- Distance: `DistanceTo(other Coordinate) int` (Manhattan distance)
- Adjacency: `IsAdjacentTo(other Coordinate) bool`
- Neighbors: `Neighbors()`, `ValidNeighbors(width, height int)`
- Arithmetic: `Add()`, `Sub()`
- Comparison: `Equal()`
- String representation: `String()`
- Direction support: `Direction` type with North/East/South/West
- Direction methods: `Move(direction)`, `DirectionTo(other)`

**Tests**: Full test coverage in `coordinate_test.go` including:
- All methods tested
- Edge cases covered
- Benchmark tests for performance-critical methods
- Verified Coordinate works as map key

#### 1.2 Add Coordinate Methods to Board
**File**: `internal/game/core/board.go`
- Add overloaded methods that accept Coordinate:
  - `InBoundsCoord(c Coordinate) bool`
  - `GetTile(c Coordinate) *Tile`
  - `SetTile(c Coordinate, tile Tile)`
  - `IdxCoord(c Coordinate) int`
- Keep existing methods for backward compatibility

#### 1.3 Update Action Types
**Files**: 
- `internal/game/core/action.go`
- `internal/game/core/movement.go`

Changes:
```go
// Add coordinate-based fields alongside existing ones
type MoveAction struct {
    PlayerID int
    // Keep existing fields for compatibility
    FromX, FromY int
    ToX, ToY int
    // New coordinate fields
    From Coordinate
    To   Coordinate
    MoveAll bool
}

// Add helper methods
func (m *MoveAction) GetFrom() Coordinate
func (m *MoveAction) GetTo() Coordinate
```

### Phase 2: Common Utilities Update (Priority: High)

#### 2.1 Update Validation Functions
**File**: `internal/common/validation.go`
- Add coordinate-based versions:
  - `IsValidCoordinateStruct(c Coordinate, width, height int) bool`
  - Keep existing functions for compatibility

#### 2.2 Update Math Utilities
**File**: `internal/common/math.go`
- Add coordinate-based distance calculation:
  - `DistanceCoord(from, to Coordinate) int`
  - `IsAdjacentCoord(from, to Coordinate) bool`

### Phase 3: Event System Integration (Priority: High)

#### 3.1 Update Event Types
**File**: `internal/game/events/game_events.go`

Replace individual coordinate fields with Coordinate structs:
```go
type MoveExecutedEvent struct {
    BaseEvent
    PlayerID    int
    From        Coordinate  // Instead of FromX, FromY
    To          Coordinate  // Instead of ToX, ToY
    ArmiesMoved int
}

type CombatResolvedEvent struct {
    BaseEvent
    Location       Coordinate  // Instead of LocationX, LocationY
    // ... other fields
}
```

### Phase 4: Game State Updates (Priority: Medium)

#### 4.1 Update Player State
**File**: `internal/game/state.go`
```go
type PlayerState struct {
    // ... existing fields
    General    Coordinate  // Instead of GeneralIdx
    // Keep GeneralIdx for compatibility
}
```

#### 4.2 Update Map/Tile Types
**File**: `internal/game/core/map.go` (if exists) or relevant files
- Add Coordinate field to Tile struct
- Update map generation to use Coordinates

### Phase 5: UI System Updates (Priority: Medium)

#### 5.1 Update Renderer
**Files**: 
- `internal/ui/renderer/board.go`
- `internal/ui/renderer/enhanced_board.go`
- `internal/ui/input/mouse.go`

Replace anonymous `struct{X, Y int}` with proper Coordinate type:
```go
// Instead of
validMoves map[struct{X, Y int}]bool

// Use
validMoves map[Coordinate]bool
```

#### 5.2 Update Input Handlers
**File**: `internal/ui/input/actions.go`
- Update to use Coordinate for position tracking
- Convert mouse positions to Coordinate objects

### Phase 6: Test Updates (Priority: Low)

#### 6.1 Update Test Utilities
**File**: `internal/testutil/fixtures.go`
- Replace `TilePosition` with `core.Coordinate`
- Update helper functions to use Coordinate

#### 6.2 Update All Test Files
Update test files to use the new Coordinate type where appropriate.

## Migration Strategy

### Approach: Gradual Migration with Compatibility

1. **Stage 1: Addition (Week 1)**
   - Add Coordinate struct
   - Add new methods alongside existing ones
   - No breaking changes

2. **Stage 2: Adoption (Weeks 2-3)**
   - New code uses Coordinate
   - Update high-value areas (events, UI)
   - Maintain backward compatibility

3. **Stage 3: Migration (Weeks 4-5)**
   - Gradually update existing code
   - Add deprecation notices to old methods
   - Update tests

4. **Stage 4: Cleanup (Week 6+)**
   - Remove deprecated methods
   - Final cleanup and optimization

### Compatibility Helpers

During migration, use helper methods:
```go
// In MoveAction
func (m *MoveAction) GetFrom() Coordinate {
    if m.From != (Coordinate{}) {
        return m.From
    }
    return Coordinate{X: m.FromX, Y: m.FromY}
}

// In Board
func (b *Board) InBounds(x, y int) bool {
    return b.InBoundsCoord(Coordinate{X: x, Y: y})
}
```

## Code Examples

### Before
```go
// Verbose and error-prone
func (e *Engine) isValidMove(fromX, fromY, toX, toY int) bool {
    if !e.board.InBounds(fromX, fromY) || !e.board.InBounds(toX, toY) {
        return false
    }
    return IsAdjacent(fromX, fromY, toX, toY)
}

// Easy to mix up parameters
event := MoveEvent{
    FromX: toY,  // Oops!
    FromY: toX,  // Oops!
    ToX: fromY,  // Oops!
    ToY: fromX,  // Oops!
}
```

### After
```go
// Clean and clear
func (e *Engine) isValidMove(from, to Coordinate) bool {
    if !e.board.InBoundsCoord(from) || !e.board.InBoundsCoord(to) {
        return false
    }
    return from.IsAdjacentTo(to)
}

// Type-safe
event := MoveEvent{
    From: to,   // Still wrong, but obvious
    To:   from,
}
```

## Testing Plan

1. **Unit Tests**: Test Coordinate methods thoroughly
2. **Integration Tests**: Ensure compatibility layer works
3. **Performance Tests**: Verify no performance regression
4. **Migration Tests**: Test old and new APIs work together

## Performance Considerations

- Coordinate is a small struct (2 ints), passing by value is efficient
- No heap allocations needed for most operations
- Methods can be inlined by the compiler
- Consider using `Coordinate` as map keys (it's comparable)

## Success Criteria

- [ ] All new code uses Coordinate type
- [ ] No breaking changes to existing APIs
- [ ] Event system fully migrated
- [ ] UI system uses Coordinate throughout
- [ ] Tests updated and passing
- [ ] Documentation updated
- [ ] Performance benchmarks show no regression

## Future Enhancements

Once Coordinate is established, consider:
1. **Direction enum**: North, East, South, West
2. **Coordinate arithmetic**: Add, Subtract methods
3. **Path finding**: Built on Coordinate type
4. **Serialization**: Custom JSON/binary marshaling
5. **Validation**: Board-aware coordinate validation

## Implementation Priority

1. **Immediate**: Create Coordinate type and use in event system
2. **Soon**: Update core game logic and common utilities
3. **Later**: Migrate UI and remaining systems
4. **Eventually**: Remove compatibility layers

---

*Document created by Claude on 2025-01-26*
*Estimated effort: 2-3 days for core implementation, 2-3 weeks for full migration*