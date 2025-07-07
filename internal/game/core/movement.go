package core

// CaptureDetails provides rich information about a capture event.
// It is returned by ApplyMoveAction if a tile changes ownership.
type CaptureDetails struct {
	X                 int // X coordinate of the captured tile
	Y                 int // Y coordinate of the captured tile
	TileType          int // Type of the tile that was captured (e.g., TileNormal, TileCity, TileGeneral)
	CapturingPlayerID int // ID of the player who made the capture and now owns the tile
	PreviousOwnerID   int // ID of the player who owned the tile before capture (can be NeutralID)
	PreviousArmyCount int // Army count on the tile before capture
}

// ApplyMoveAction applies a move action to the board.
// Returns CaptureDetails if the move resulted in a tile changing ownership, otherwise nil.
// An error is returned if the action is invalid or an issue occurs.
// The changedTiles parameter is optional - if provided, modified tile indices will be added to it.
func ApplyMoveAction(b *Board, action *MoveAction, changedTiles map[int]struct{}) (*CaptureDetails, error) {
	// Validate the action first (this now includes checks for target tile type like mountains)
	if err := action.Validate(b, action.PlayerID); err != nil {
		return nil, err
	}

	fromIdx := b.Idx(action.FromX, action.FromY)
	toIdx := b.Idx(action.ToX, action.ToY)

	fromTile := &b.T[fromIdx]
	toTile := &b.T[toIdx]

	// Store pre-capture state of toTile for CaptureDetails
	originalToTileOwner := toTile.Owner
	originalToTileArmy := toTile.Army
	// TileType of toTile does not change upon capture, only owner and army.

	var armiesToMove int
	if action.MoveAll {
		armiesToMove = fromTile.Army - 1
	} else {
		// Move half (integer division), minimum 1
		armiesToMove = fromTile.Army / 2
		if armiesToMove == 0 {
			armiesToMove = 1
		}
	}


	var captureDetails *CaptureDetails = nil

	// Apply the army reduction from the source tile
	fromTile.Army -= armiesToMove
	
	// Track changed tiles if map provided
	if changedTiles != nil {
		changedTiles[fromIdx] = struct{}{}
		changedTiles[toIdx] = struct{}{}
	}

	if toTile.Owner == action.PlayerID {
		// Fast path: Moving to own tile - just consolidate armies
		toTile.Army += armiesToMove
		return nil, nil
	}

	// Combat path: Moving to an enemy or neutral tile
	if armiesToMove > toTile.Army {
		// Successful capture - tile changes ownership
		toTile.Owner = action.PlayerID
		toTile.Army = armiesToMove - toTile.Army

		captureDetails = &CaptureDetails{
			X:                 action.ToX,
			Y:                 action.ToY,
			TileType:          toTile.Type,
			CapturingPlayerID: action.PlayerID,
			PreviousOwnerID:   originalToTileOwner,
			PreviousArmyCount: originalToTileArmy,
		}
	} else {
		// Failed attack - defender loses armies but retains ownership
		toTile.Army -= armiesToMove
	}

	return captureDetails, nil
}

// PlayerEliminationOrder instructs the engine on which player was eliminated
// and who should take over their tiles.
type PlayerEliminationOrder struct {
	EliminatedPlayerID int
	NewOwnerID         int // ID of the player who captured the general and takes over tiles
}

// ProcessCaptures analyzes all capture events from a turn and identifies
// player eliminations based on General captures.
func ProcessCaptures(allCaptureEventsThisTurn []CaptureDetails) []PlayerEliminationOrder {
	var eliminationOrders []PlayerEliminationOrder
	eliminationProcessedFor := make(map[int]bool)

	for _, capture := range allCaptureEventsThisTurn {
		if capture.TileType == TileGeneral &&
			capture.PreviousOwnerID != NeutralID &&
			capture.PreviousOwnerID != capture.CapturingPlayerID &&
			!eliminationProcessedFor[capture.PreviousOwnerID] {

			eliminationOrders = append(eliminationOrders, PlayerEliminationOrder{
				EliminatedPlayerID: capture.PreviousOwnerID,
				NewOwnerID:         capture.CapturingPlayerID,
			})
			eliminationProcessedFor[capture.PreviousOwnerID] = true
		}
	}
	return eliminationOrders
}