package rules

import "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"

// LegalMoveCalculator computes legal moves for players
type LegalMoveCalculator struct{}

// NewLegalMoveCalculator creates a new legal move calculator
func NewLegalMoveCalculator() *LegalMoveCalculator {
	return &LegalMoveCalculator{}
}

// GetLegalActionMask returns a flattened boolean mask indicating which actions are legal for the given player.
// For a board of width W and height H:
// - Total actions = W * H * 4 (4 directions per tile)
// - Index = (y * W + x) * 4 + direction
// - Directions: 0=up, 1=right, 2=down, 3=left
// - true = legal move, false = illegal move
func (lmc *LegalMoveCalculator) GetLegalActionMask(board *core.Board, player Player, ownedTiles []int) []bool {
	width := board.W
	height := board.H
	maskSize := width * height * 4
	mask := make([]bool, maskSize)

	// If player is not alive, return all false
	if !player.IsAlive() {
		return mask
	}

	playerID := player.GetID()

	// Direction offsets: up, right, down, left
	dx := []int{0, 1, 0, -1}
	dy := []int{-1, 0, 1, 0}

	// Check each tile the player owns
	for _, tileIdx := range ownedTiles {
		tile := &board.T[tileIdx]

		// Can only move if we own the tile and have more than 1 army
		if tile.Owner != playerID || tile.Army <= 1 {
			continue
		}

		// Get x,y coordinates from tile index
		x := tileIdx % width
		y := tileIdx / width

		// Check each direction
		for dir := 0; dir < 4; dir++ {
			newX := x + dx[dir]
			newY := y + dy[dir]

			// Create a move action to validate
			moveAction := &core.MoveAction{
				PlayerID: playerID,
				FromX:    x,
				FromY:    y,
				ToX:      newX,
				ToY:      newY,
			}

			// Use the existing validation logic
			if err := moveAction.Validate(board, playerID); err == nil {
				// This is a legal move
				actionIdx := (y*width+x)*4 + dir
				mask[actionIdx] = true
			}
		}
	}

	return mask
}
