package game

import "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"

// This file contains the optimized bitfield-based visibility implementation.
// It will be used when config.Features.UseOptimizedVisibility is true.

// Precomputed offsets for 3x3 visibility pattern
var visibilityOffsets = []struct{ dx, dy int }{
	{-1, -1}, {0, -1}, {1, -1},
	{-1, 0}, {0, 0}, {1, 0},
	{-1, 1}, {0, 1}, {1, 1},
}

// updateFogOfWarOptimized updates fog of war using bitfield operations
func (e *Engine) updateFogOfWarOptimized() {
	if !e.gs.FogOfWarEnabled {
		return
	}

	// On turn 0 or if too many visibility changes, do full update
	fullUpdateThreshold := len(e.gs.Board.T) / 10 // 10% of board
	if e.gs.Turn == 0 || len(e.gs.VisibilityChangedTiles) > fullUpdateThreshold {
		e.performFullVisibilityUpdateOptimized()
		return
	}

	// Incremental update based on changed tiles
	e.performIncrementalVisibilityUpdateOptimized()
}

// performFullVisibilityUpdateOptimized recalculates all visibility using bitfields
func (e *Engine) performFullVisibilityUpdateOptimized() {
	// Clear all visibility bitfields
	for i := range e.gs.Board.T {
		e.gs.Board.T[i].VisibleBitfield = 0
	}

	// Set visibility from all owned tiles
	for pid, p := range e.gs.Players {
		if !p.Alive {
			continue
		}
		playerBit := uint32(1 << uint(pid))
		for _, tileIdx := range p.OwnedTiles {
			e.setVisibilityAroundOptimized(tileIdx, pid, playerBit)
		}
	}

	// Sync to map for compatibility
	for i := range e.gs.Board.T {
		e.gs.Board.T[i].SyncVisibilityToMap()
	}
	
	e.logger.Debug().Msg("Performed full visibility update (optimized)")
}

// performIncrementalVisibilityUpdateOptimized updates visibility for changed tiles only
func (e *Engine) performIncrementalVisibilityUpdateOptimized() {
	// Track which tiles need visibility sync
	tilesToSync := make(map[int]struct{})

	// For each tile that changed ownership
	for tileIdx := range e.gs.VisibilityChangedTiles {
		if tileIdx < 0 || tileIdx >= len(e.gs.Board.T) {
			continue
		}
		
		// Clear visibility around this tile
		e.clearVisibilityAroundOptimized(tileIdx, tilesToSync)
		
		// If tile has a new owner, grant visibility
		currentOwner := e.gs.Board.T[tileIdx].Owner
		if currentOwner >= 0 && currentOwner < len(e.gs.Players) && e.gs.Players[currentOwner].Alive {
			playerBit := uint32(1 << uint(currentOwner))
			e.setVisibilityAroundOptimized(tileIdx, currentOwner, playerBit)
			// Mark affected tiles for sync
			e.markVisibilityTilesForSync(tileIdx, tilesToSync)
		}
	}

	// Re-establish visibility from all owned tiles of affected players
	for k := range e.tempAffectedPlayers {
		delete(e.tempAffectedPlayers, k)
	}
	
	for tileIdx := range e.gs.VisibilityChangedTiles {
		tile := &e.gs.Board.T[tileIdx]
		if tile.Owner >= 0 {
			e.tempAffectedPlayers[tile.Owner] = struct{}{}
		}
	}
	
	for pid := range e.tempAffectedPlayers {
		if !e.gs.Players[pid].Alive {
			continue
		}
		playerBit := uint32(1 << uint(pid))
		for _, ownedTileIdx := range e.gs.Players[pid].OwnedTiles {
			e.setVisibilityAroundOptimized(ownedTileIdx, pid, playerBit)
			e.markVisibilityTilesForSync(ownedTileIdx, tilesToSync)
		}
	}

	// Sync only affected tiles to map for compatibility
	for tileIdx := range tilesToSync {
		e.gs.Board.T[tileIdx].SyncVisibilityToMap()
	}
	
	e.logger.Debug().Int("changed_tiles", len(e.gs.VisibilityChangedTiles)).Msg("Performed incremental visibility update (optimized)")
}

// setVisibilityAroundOptimized sets visibility using bitfield operations
func (e *Engine) setVisibilityAroundOptimized(tileIdx int, playerID int, playerBit uint32) {
	x, y := e.gs.Board.XY(tileIdx)
	
	for _, offset := range visibilityOffsets {
		nx, ny := x + offset.dx, y + offset.dy
		if e.gs.Board.InBounds(nx, ny) {
			visIdx := e.gs.Board.Idx(nx, ny)
			e.gs.Board.T[visIdx].VisibleBitfield |= playerBit
		}
	}
}

// clearVisibilityAroundOptimized clears visibility using bitfield operations
func (e *Engine) clearVisibilityAroundOptimized(tileIdx int, tilesToSync map[int]struct{}) {
	x, y := e.gs.Board.XY(tileIdx)
	
	// Create bitmask of all players
	allPlayersMask := uint32(0)
	for pid := range e.gs.Players {
		allPlayersMask |= (1 << uint(pid))
	}
	clearMask := ^allPlayersMask
	
	for _, offset := range visibilityOffsets {
		nx, ny := x + offset.dx, y + offset.dy
		if e.gs.Board.InBounds(nx, ny) {
			visIdx := e.gs.Board.Idx(nx, ny)
			e.gs.Board.T[visIdx].VisibleBitfield &= clearMask
			tilesToSync[visIdx] = struct{}{}
		}
	}
}

// markVisibilityTilesForSync marks tiles that need map sync
func (e *Engine) markVisibilityTilesForSync(tileIdx int, tilesToSync map[int]struct{}) {
	x, y := e.gs.Board.XY(tileIdx)
	
	for _, offset := range visibilityOffsets {
		nx, ny := x + offset.dx, y + offset.dy
		if e.gs.Board.InBounds(nx, ny) {
			visIdx := e.gs.Board.Idx(nx, ny)
			tilesToSync[visIdx] = struct{}{}
		}
	}
}

// ComputePlayerVisibilityOptimized computes visibility using bitfields
func (e *Engine) ComputePlayerVisibilityOptimized(playerID int) PlayerVisibility {
	numTiles := len(e.gs.Board.T)
	vis := PlayerVisibility{
		VisibleTiles: make([]bool, numTiles),
		FogTiles:     make([]bool, numTiles),
	}
	
	// If fog of war is disabled, all tiles are visible
	if !e.gs.FogOfWarEnabled {
		for i := range vis.VisibleTiles {
			vis.VisibleTiles[i] = true
		}
		return vis
	}
	
	// Use bitfield for visibility check
	playerBit := uint32(1 << uint(playerID))
	
	// Copy current visibility from tiles
	for i, tile := range e.gs.Board.T {
		vis.VisibleTiles[i] = (tile.VisibleBitfield & playerBit) != 0
		
		// A tile is in fog if it was previously discovered but not currently visible
		if !vis.VisibleTiles[i] && tile.Type != core.TileNormal {
			vis.FogTiles[i] = true
		}
	}
	
	return vis
}