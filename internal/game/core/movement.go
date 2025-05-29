package core

// ApplyMoveAction applies a move action to the board
// Returns true if the move resulted in a capture
func ApplyMoveAction(b *Board, action *MoveAction) (bool, error) {
	if err := action.Validate(b, action.PlayerID); err != nil {
		return false, err
	}
	
	fromIdx := b.Idx(action.FromX, action.FromY)
	toIdx := b.Idx(action.ToX, action.ToY)
	
	fromTile := &b.T[fromIdx]
	toTile := &b.T[toIdx]
	
	// Calculate armies to move
	var armiesToMove int
	if action.MoveAll {
		armiesToMove = fromTile.Army - 1 // Leave 1 behind
	} else {
		armiesToMove = fromTile.Army / 2 // Move half (rounded down)
		if armiesToMove == 0 {
			armiesToMove = 1 // Always move at least 1 if we have >1
		}
	}
	
	// Apply the move
	fromTile.Army -= armiesToMove
	
	captured := false
	if toTile.Owner == action.PlayerID {
		// Moving to own tile - just add armies
		toTile.Army += armiesToMove
	} else {
		// Combat resolution
		if armiesToMove > toTile.Army {
			// Successful capture
			toTile.Owner = action.PlayerID
			toTile.Army = armiesToMove - toTile.Army
			captured = true
		} else {
			// Failed attack
			toTile.Army -= armiesToMove
		}
	}
	
	return captured, nil
}

// ProcessCaptures handles cascading captures (cities/generals)
func ProcessCaptures(b *Board, captures []CaptureInfo) []CaptureInfo {
	var newCaptures []CaptureInfo
	
	for _, capture := range captures {
		idx := b.Idx(capture.X, capture.Y)
		tile := &b.T[idx]
		
		if tile.IsGeneral() || tile.IsCity() {
			// Special handling for important tiles
			newCaptures = append(newCaptures, capture)
		}
	}
	
	return newCaptures
}

// CaptureInfo represents information about a captured tile
type CaptureInfo struct {
	X, Y     int
	PlayerID int
	TileType int
}