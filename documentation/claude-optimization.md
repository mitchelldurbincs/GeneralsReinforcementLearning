# Visibility System Optimization Action Plan

## Overview
The current visibility system uses `map[int]bool` for each tile (128+ bytes overhead), causing significant memory allocation and GC pressure. We'll replace this with a uint32 bitfield (4 bytes) while maintaining identical functionality.

## Phase 1: Data Structure Changes (Day 1)

### 1. Update Tile Structure
- Change `Visible map[int]bool` to `Visible uint32` in `core/tile.go`
- Add helper methods: `IsVisibleTo(playerID)`, `SetVisible(playerID, bool)`
- Support up to 32 players (current max is 4, so plenty of headroom)

### 2. Add Compatibility Layer
- Temporary methods to convert between map and bitfield representations
- Allows gradual migration without breaking existing code

## Phase 2: Algorithm Optimizations (Day 2-3)

### 1. Precompute Visibility Patterns
- Create lookup table for 3x3 visibility offsets
- Eliminate repeated coordinate calculations

### 2. Implement Delta Tracking
- Track only tiles where ownership changed
- Maintain `previousOwners` map for efficient removal
- Skip visibility updates for unchanged areas

### 3. Optimize Update Operations
- Replace nested loops with bitwise operations
- Use single pass updates instead of clear-then-add pattern
- Batch visibility changes before applying

## Phase 3: Integration (Day 4)

### 1. Update All Consumers
- `rendering.go`: Use `IsVisibleTo()` instead of map access
- `grpc/gameserver`: Update visibility checks
- `ui/renderer`: Update fog of war rendering

### 2. Remove Old Code
- Delete map-based visibility code
- Remove compatibility layer
- Clean up unused imports

## Phase 4: Testing & Validation (Day 5)

### 1. Unit Tests
- Bitfield operations correctness
- Boundary conditions
- Multi-player scenarios

### 2. Integration Tests
- Compare output with original implementation
- Verify fog of war behavior unchanged
- Test with various player counts and map sizes

### 3. Performance Benchmarks
- Memory allocation reduction
- CPU time improvement
- GC pressure analysis

## Implementation Details

### New Tile Structure
```go
type Tile struct {
    Owner   int
    Army    int
    Type    TileType
    Visible uint32  // Bit i = 1 if player i can see this tile
}
```

### Helper Methods
```go
// Check if tile is visible to a player
func (t *Tile) IsVisibleTo(playerID int) bool {
    return t.Visible & (1 << playerID) != 0
}

// Set visibility for a player
func (t *Tile) SetVisible(playerID int, visible bool) {
    if visible {
        t.Visible |= (1 << playerID)
    } else {
        t.Visible &^= (1 << playerID)
    }
}
```

### Visibility Update Algorithm
```go
// Precomputed offsets for 3x3 visibility
var visibilityOffsets = []struct{dx, dy int}{
    {-1,-1}, {0,-1}, {1,-1},
    {-1, 0}, {0, 0}, {1, 0},
    {-1, 1}, {0, 1}, {1, 1},
}

// Add visibility around a tile
func addVisibilityAround(visibility []uint32, tileIdx int, playerID int) {
    x, y := board.XY(tileIdx)
    playerBit := uint32(1 << playerID)
    
    for _, offset := range visibilityOffsets {
        nx, ny := x + offset.dx, y + offset.dy
        if board.InBounds(nx, ny) {
            idx := board.Idx(nx, ny)
            visibility[idx] |= playerBit
        }
    }
}
```

## Key Optimizations
- **Memory**: 32x reduction per tile (128 bytes → 4 bytes)
- **CPU**: O(1) visibility checks vs O(n) map lookups
- **GC**: Zero allocations for visibility updates
- **Cache**: Better locality with contiguous memory

## Success Metrics
- ✓ Memory usage reduced by >90% for visibility data
- ✓ Turn processing time reduced by 50-70% on large maps
- ✓ Zero functional changes (all tests pass)
- ✓ No visual differences in game

## Rollback Plan
- Keep original implementation behind feature flag
- A/B test with subset of games
- Monitor performance metrics
- One-line config change to revert

## Risk Assessment

### Low Risk
- Bitfield operations are well-understood and reliable
- Changes are localized to visibility system
- Backward compatibility can be maintained during migration

### Medium Risk
- Need to ensure all 32 player positions work correctly
- Edge cases with board boundaries need careful testing
- Performance characteristics may change for small player counts

### Mitigation Strategies
1. **Feature Flag**: Add `UseOptimizedVisibility` flag to switch between implementations
2. **Extensive Testing**: Full test coverage before deployment
3. **Gradual Rollout**: Test with small games first, then larger ones
4. **Monitoring**: Add metrics for visibility update performance
5. **Rollback Plan**: Keep old implementation available for quick revert

## Validation Checklist
- [ ] All unit tests pass
- [ ] Integration tests show identical behavior
- [ ] Benchmarks show improved performance
- [ ] Memory usage is reduced
- [ ] No regression in game functionality
- [ ] UI rendering works correctly
- [ ] Fog of war toggle still works
- [ ] Edge cases (board boundaries, dead players) handled