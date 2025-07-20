package game

import "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"

// This file contains all fog of war and visibility-related functionality for the game engine.

// updateFogOfWar updates the fog of war visibility for all players
func (e *Engine) updateFogOfWar() {
	if !e.gs.FogOfWarEnabled {
		return
	}

	// On turn 0 or if too many visibility changes, do full update
	fullUpdateThreshold := len(e.gs.Board.T) / 10 // 10% of board
	if e.gs.Turn == 0 || len(e.gs.VisibilityChangedTiles) > fullUpdateThreshold {
		e.performFullVisibilityUpdate()
		return
	}

	// Incremental update based on changed tiles
	e.performIncrementalVisibilityUpdate()
}

// performFullVisibilityUpdate recalculates all visibility from scratch
func (e *Engine) performFullVisibilityUpdate() {
	// Clear all visibility
	for i := range e.gs.Board.T {
		for pid := range e.gs.Players {
			e.gs.Board.T[i].Visible[pid] = false
		}
	}

	// Set visibility from all owned tiles
	for pid, p := range e.gs.Players {
		if !p.Alive {
			continue
		}
		for _, tileIdx := range p.OwnedTiles {
			e.setVisibilityAround(tileIdx, pid)
		}
	}
	
	e.logger.Debug().Msg("Performed full visibility update")
}

// performIncrementalVisibilityUpdate only updates visibility for changed tiles
func (e *Engine) performIncrementalVisibilityUpdate() {
	// For each tile that changed ownership, update visibility
	for tileIdx := range e.gs.VisibilityChangedTiles {
		if tileIdx < 0 || tileIdx >= len(e.gs.Board.T) {
			continue
		}
		
		// Get the current owner of the tile
		currentOwner := e.gs.Board.T[tileIdx].Owner
		
		// Clear visibility for all players around this tile first
		// This is necessary because we don't track previous owners
		e.clearVisibilityAround(tileIdx)
		
		// If tile has a new owner, grant visibility
		if currentOwner >= 0 && currentOwner < len(e.gs.Players) && e.gs.Players[currentOwner].Alive {
			e.setVisibilityAround(tileIdx, currentOwner)
		}
	}
	
	// Re-establish visibility from all owned tiles of affected players
	// This prevents removing visibility that should still exist from other tiles
	
	// Clear and reuse temporary map
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
		for _, ownedTileIdx := range e.gs.Players[pid].OwnedTiles {
			e.setVisibilityAround(ownedTileIdx, pid)
		}
	}
	
	e.logger.Debug().Int("changed_tiles", len(e.gs.VisibilityChangedTiles)).Msg("Performed incremental visibility update")
}

// setVisibilityAround sets visibility for a player in a 3x3 area around a tile
func (e *Engine) setVisibilityAround(tileIdx int, playerID int) {
	x, y := e.gs.Board.XY(tileIdx)
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			nx, ny := x+dx, y+dy
			if nx >= 0 && nx < e.gs.Board.W && ny >= 0 && ny < e.gs.Board.H {
				visIdx := e.gs.Board.Idx(nx, ny)
				e.gs.Board.T[visIdx].Visible[playerID] = true
			}
		}
	}
}

// clearVisibilityAround clears visibility for all players in a 3x3 area around a tile
func (e *Engine) clearVisibilityAround(tileIdx int) {
	x, y := e.gs.Board.XY(tileIdx)
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			nx, ny := x+dx, y+dy
			if nx >= 0 && nx < e.gs.Board.W && ny >= 0 && ny < e.gs.Board.H {
				visIdx := e.gs.Board.Idx(nx, ny)
				for pid := range e.gs.Players {
					e.gs.Board.T[visIdx].Visible[pid] = false
				}
			}
		}
	}
}

// PlayerVisibility contains visibility information for a specific player
type PlayerVisibility struct {
	VisibleTiles []bool // Currently visible tiles
	FogTiles     []bool // Previously seen but not currently visible tiles
}

// ComputePlayerVisibility computes visibility information for a specific player
func (e *Engine) ComputePlayerVisibility(playerID int) PlayerVisibility {
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
	
	// Copy current visibility from tiles
	for i, tile := range e.gs.Board.T {
		vis.VisibleTiles[i] = tile.Visible[playerID]
		
		// A tile is in fog if it was previously discovered (has a type or we've seen it before)
		// but is not currently visible
		if !vis.VisibleTiles[i] {
			// Check if we've discovered this tile before
			// For now, we'll mark as fog if it's a special tile type we would have seen
			if tile.Type != core.TileNormal {
				vis.FogTiles[i] = true
			}
		}
	}
	
	return vis
}