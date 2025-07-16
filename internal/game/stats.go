package game

// This file contains all player statistics management functionality for the game engine.

// updatePlayerStats recalculates player statistics
// On turn 0 or when no previous stats exist, it does a full scan.
// Otherwise, it uses incremental updates based on ChangedTiles.
func (e *Engine) updatePlayerStats() {
	// Only do full update if we have changed tiles or on initialization
	if len(e.gs.ChangedTiles) == 0 && e.gs.Turn > 0 {
		// No changes, stats are already up to date
		e.logger.Debug().Msg("No tile changes, skipping player stats update")
		return
	}

	e.logger.Debug().Int("changed_tiles", len(e.gs.ChangedTiles)).Msg("Updating player stats")

	// On turn 0 or if we have too many changes, do a full update
	// Using 20% of board size as threshold for full update vs incremental
	fullUpdateThreshold := len(e.gs.Board.T) / 5
	if e.gs.Turn == 0 || len(e.gs.ChangedTiles) > fullUpdateThreshold {
		e.logger.Debug().Msg("Performing full stats update")
		e.performFullStatsUpdate()
		return
	}

	// Incremental update based on changed tiles
	e.logger.Debug().Msg("Performing incremental stats update")
	e.performIncrementalStatsUpdate()
}

// performFullStatsUpdate does a complete recalculation of all player stats
func (e *Engine) performFullStatsUpdate() {
	for pid := range e.gs.Players {
		e.gs.Players[pid].ArmyCount = 0
		e.gs.Players[pid].GeneralIdx = -1
		e.gs.Players[pid].OwnedTiles = e.gs.Players[pid].OwnedTiles[:0] // Clear but keep capacity
	}

	for idx, t := range e.gs.Board.T {
		if t.Owner >= 0 && t.Owner < len(e.gs.Players) {
			p := &e.gs.Players[t.Owner]
			p.ArmyCount += t.Army
			p.OwnedTiles = append(p.OwnedTiles, idx)
			if t.IsGeneral() {
				p.GeneralIdx = idx
			}
		}
	}

	// Update alive status based on general ownership
	for pid := range e.gs.Players {
		oldAliveStatus := e.gs.Players[pid].Alive
		e.gs.Players[pid].Alive = e.gs.Players[pid].GeneralIdx != -1
		if oldAliveStatus && !e.gs.Players[pid].Alive {
			e.logger.Info().Int("player_id", pid).Msg("Player lost their general and is now marked as dead")
		} else if !oldAliveStatus && e.gs.Players[pid].Alive {
			// This case should ideally not happen unless a general is respawned/reassigned.
			e.logger.Warn().Int("player_id", pid).Msg("Player was dead and is now marked as alive - unexpected general appearance?")
		}
	}
	e.logger.Debug().Msg("Player stats updated")
}

// performIncrementalStatsUpdate updates stats based only on changed tiles
func (e *Engine) performIncrementalStatsUpdate() {
	// First pass: subtract old ownership from stats using our temporary map
	// Clear and reuse temporary map
	for k := range e.tempTileOwnership {
		delete(e.tempTileOwnership, k)
	}
	
	// Scan changed tiles and build ownership map
	for tileIdx := range e.gs.ChangedTiles {
		tile := &e.gs.Board.T[tileIdx]
		e.tempTileOwnership[tileIdx] = tile.Owner
	}
	
	// Second pass: rebuild OwnedTiles for affected players
	affectedPlayers := make(map[int]bool)
	for _, currentOwner := range e.tempTileOwnership {
		if currentOwner >= 0 {
			affectedPlayers[currentOwner] = true
		}
		// We don't know previous owners, so we need to rebuild all players' OwnedTiles
	}
	
	// For simplicity in incremental update, rebuild stats for all players
	// This is still more efficient than scanning all tiles
	for pid := range e.gs.Players {
		e.gs.Players[pid].ArmyCount = 0
		e.gs.Players[pid].GeneralIdx = -1
		newOwnedTiles := e.gs.Players[pid].OwnedTiles[:0] // Reuse backing array
		
		// Re-scan only this player's previously owned tiles
		for _, tileIdx := range e.gs.Players[pid].OwnedTiles {
			if e.gs.Board.T[tileIdx].Owner == pid {
				tile := &e.gs.Board.T[tileIdx]
				e.gs.Players[pid].ArmyCount += tile.Army
				newOwnedTiles = append(newOwnedTiles, tileIdx)
				if tile.IsGeneral() {
					e.gs.Players[pid].GeneralIdx = tileIdx
				}
			}
		}
		
		// Add any newly owned tiles
		for tileIdx, owner := range e.tempTileOwnership {
			if owner == pid {
				// Check if we already processed this tile
				found := false
				for _, existingIdx := range newOwnedTiles {
					if existingIdx == tileIdx {
						found = true
						break
					}
				}
				if !found {
					tile := &e.gs.Board.T[tileIdx]
					e.gs.Players[pid].ArmyCount += tile.Army
					newOwnedTiles = append(newOwnedTiles, tileIdx)
					if tile.IsGeneral() {
						e.gs.Players[pid].GeneralIdx = tileIdx
					}
				}
			}
		}
		
		e.gs.Players[pid].OwnedTiles = newOwnedTiles
	}
	
	// Update alive status
	for pid := range e.gs.Players {
		oldAliveStatus := e.gs.Players[pid].Alive
		e.gs.Players[pid].Alive = e.gs.Players[pid].GeneralIdx != -1
		if oldAliveStatus && !e.gs.Players[pid].Alive {
			e.logger.Info().Int("player_id", pid).Msg("Player lost their general and is now marked as dead")
		} else if !oldAliveStatus && e.gs.Players[pid].Alive {
			// This case should ideally not happen unless a general is respawned/reassigned.
			e.logger.Warn().Int("player_id", pid).Msg("Player was dead and is now marked as alive - unexpected general appearance?")
		}
	}
	e.logger.Debug().Msg("Player stats updated")
}