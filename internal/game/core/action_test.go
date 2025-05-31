package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Helper to create a board with a default setup for some tests
func setupBoardForActionTests(w, h int) *Board {
	board := NewBoard(w, h)
	// Example: Player 0 owns tile (1,1) with 10 armies
	// Adjust as needed for specific tests, or setup in each test
	if w > 1 && h > 1 {
		idx := board.Idx(1, 1)
		board.T[idx].Owner = 0
		board.T[idx].Army = 10
		board.T[idx].Type = TileNormal
	}
	return board
}

func TestMoveAction_Validate(t *testing.T) {
	playerID := 0
	boardWidth := 5
	boardHeight := 5

	// Test Case: Valid Move
	t.Run("ValidMove", func(t *testing.T) {
		board := NewBoard(boardWidth, boardHeight)
		// Setup FromTile: Player 0, (1,1), 5 armies, Normal
		fromIdx := board.Idx(1, 1)
		board.T[fromIdx].Owner = playerID
		board.T[fromIdx].Army = 5
		board.T[fromIdx].Type = TileNormal
		// Setup ToTile: Neutral, (1,2), Normal
		toIdx := board.Idx(1, 2)
		board.T[toIdx].Owner = NeutralID
		board.T[toIdx].Type = TileNormal

		action := MoveAction{
			PlayerID: playerID,
			FromX:    1, FromY: 1,
			ToX:      1, ToY: 2,
			MoveAll:  true,
		}
		err := action.Validate(board, playerID)
		assert.NoError(t, err, "Valid move should not produce an error")
	})

	// Test Cases: Invalid Coordinates
	invalidCoordTests := []struct {
		name          string
		action        MoveAction
		expectedError error
	}{
		{"FromXNegative", MoveAction{PlayerID: playerID, FromX: -1, FromY: 1, ToX: 0, ToY: 1}, ErrInvalidCoordinates},
		{"FromXTooLarge", MoveAction{PlayerID: playerID, FromX: boardWidth, FromY: 1, ToX: 0, ToY: 1}, ErrInvalidCoordinates},
		{"FromYNegative", MoveAction{PlayerID: playerID, FromX: 1, FromY: -1, ToX: 1, ToY: 0}, ErrInvalidCoordinates},
		{"FromYTooLarge", MoveAction{PlayerID: playerID, FromX: 1, FromY: boardHeight, ToX: 1, ToY: 0}, ErrInvalidCoordinates},
		{"ToXNegative", MoveAction{PlayerID: playerID, FromX: 1, FromY: 1, ToX: -1, ToY: 1}, ErrInvalidCoordinates},
		{"ToXTooLarge", MoveAction{PlayerID: playerID, FromX: 1, FromY: 1, ToX: boardWidth, ToY: 1}, ErrInvalidCoordinates},
		{"ToYNegative", MoveAction{PlayerID: playerID, FromX: 1, FromY: 1, ToX: 1, ToY: -1}, ErrInvalidCoordinates},
		{"ToYTooLarge", MoveAction{PlayerID: playerID, FromX: 1, FromY: 1, ToX: 1, ToY: boardHeight}, ErrInvalidCoordinates},
	}

	for _, tc := range invalidCoordTests {
		t.Run("InvalidCoordinates_"+tc.name, func(t *testing.T) {
			board := NewBoard(boardWidth, boardHeight) // Fresh board for each coord test
			// Minimal setup for FromTile to pass other checks if coords were valid
			if tc.action.FromX >= 0 && tc.action.FromX < boardWidth && tc.action.FromY >= 0 && tc.action.FromY < boardHeight {
				board.T[board.Idx(tc.action.FromX, tc.action.FromY)].Owner = playerID
				board.T[board.Idx(tc.action.FromX, tc.action.FromY)].Army = 5
			}
			err := tc.action.Validate(board, playerID)
			assert.ErrorIs(t, err, tc.expectedError, "Expected ErrInvalidCoordinates")
		})
	}

	// Test Case: Move to Self
	t.Run("MoveToSelf", func(t *testing.T) {
		board := setupBoardForActionTests(boardWidth, boardHeight) // Uses default (1,1) owned by P0
		action := MoveAction{
			PlayerID: playerID,
			FromX:    1, FromY: 1,
			ToX:      1, ToY: 1, // Moving to self
		}
		err := action.Validate(board, playerID)
		assert.ErrorIs(t, err, ErrMoveToSelf, "Expected ErrMoveToSelf")
	})

	// Test Case: Not Adjacent
	t.Run("NotAdjacent", func(t *testing.T) {
		board := setupBoardForActionTests(boardWidth, boardHeight)
		action := MoveAction{
			PlayerID: playerID,
			FromX:    1, FromY: 1,
			ToX:      3, ToY: 3, // Diagonal and not adjacent
		}
		err := action.Validate(board, playerID)
		assert.ErrorIs(t, err, ErrNotAdjacent, "Expected ErrNotAdjacent")
	})
    
    	t.Run("NotAdjacentButSameRowFar", func(t *testing.T) {
		board := setupBoardForActionTests(boardWidth, boardHeight)
		action := MoveAction{
			PlayerID: playerID,
			FromX:    1, FromY: 1,
			ToX:      3, ToY: 1, // Same row, but not adjacent
		}
		err := action.Validate(board, playerID)
		assert.ErrorIs(t, err, ErrNotAdjacent, "Expected ErrNotAdjacent for same row, non-adjacent")
	})


	// Test Case: Not Owned
	t.Run("NotOwned", func(t *testing.T) {
		board := NewBoard(boardWidth, boardHeight)
		// FromTile is owned by Player 1, action by Player 0
		fromIdx := board.Idx(1, 1)
		board.T[fromIdx].Owner = playerID + 1 // Different owner
		board.T[fromIdx].Army = 10
		board.T[fromIdx].Type = TileNormal

		toIdx := board.Idx(1, 2) // Valid target tile
		board.T[toIdx].Type = TileNormal


		action := MoveAction{
			PlayerID: playerID, // Player 0 trying to move
			FromX:    1, FromY: 1,
			ToX:      1, ToY: 2,
		}
		err := action.Validate(board, playerID)
		assert.ErrorIs(t, err, ErrNotOwned, "Expected ErrNotOwned")
	})

	// Test Cases: Insufficient Army
	insufficientArmyTests := []struct {
		name      string
		armyCount int
	}{
		{"ArmyIs1", 1},
		{"ArmyIs0", 0},
	}
	for _, tc := range insufficientArmyTests {
		t.Run("InsufficientArmy_"+tc.name, func(t *testing.T) {
			board := NewBoard(boardWidth, boardHeight)
			fromIdx := board.Idx(1, 1)
			board.T[fromIdx].Owner = playerID
			board.T[fromIdx].Army = tc.armyCount // Set insufficient army
			board.T[fromIdx].Type = TileNormal

			toIdx := board.Idx(1, 2) // Valid target tile
            board.T[toIdx].Type = TileNormal


			action := MoveAction{
				PlayerID: playerID,
				FromX:    1, FromY: 1,
				ToX:      1, ToY: 2,
			}
			err := action.Validate(board, playerID)
			assert.ErrorIs(t, err, ErrInsufficientArmy, "Expected ErrInsufficientArmy")
		})
	}

	// Test Case: Target is Mountain
	t.Run("TargetIsMountain", func(t *testing.T) {
		board := NewBoard(boardWidth, boardHeight)
		// Setup FromTile
		fromIdx := board.Idx(1, 1)
		board.T[fromIdx].Owner = playerID
		board.T[fromIdx].Army = 10
		board.T[fromIdx].Type = TileNormal

		// Setup ToTile as Mountain
		toIdx := board.Idx(1, 2)
		board.T[toIdx].Owner = NeutralID
		board.T[toIdx].Type = TileMountain // Target is a mountain

		action := MoveAction{
			PlayerID: playerID,
			FromX:    1, FromY: 1,
			ToX:      1, ToY: 2,
		}
		err := action.Validate(board, playerID)
		assert.ErrorIs(t, err, ErrTargetIsMountain, "Expected ErrTargetIsMountain")
	})
}