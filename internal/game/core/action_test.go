package core

import (
	"testing"
	"github.com/stretchr/testify/assert"
)

// Test naming convention: TestFunction_Scenario_ExpectedBehavior
func TestMoveAction_ValidateBasicMove_Success(t *testing.T) {
	// Arrange
	board := NewBoard(5, 5)
	board.T[board.Idx(1, 1)] = Tile{Owner: 0, Army: 5, Type: TileNormal}
	
	action := &MoveAction{
		PlayerID: 0,
		FromX:    1, FromY: 1,
		ToX:      1, ToY: 2,
		MoveAll:  true,
	}
	
	// Act
	err := action.Validate(board, 0)
	
	// Assert
	assert.NoError(t, err, "Valid move should not return error")
}

func TestMoveAction_ValidateInsufficientArmy_ReturnsError(t *testing.T) {
	// Arrange
	board := NewBoard(5, 5)
	board.T[board.Idx(1, 1)] = Tile{Owner: 0, Army: 1, Type: TileNormal} // Only 1 army
	
	action := &MoveAction{
		PlayerID: 0,
		FromX:    1, FromY: 1,
		ToX:      1, ToY: 2,
		MoveAll:  true,
	}
	
	// Act
	err := action.Validate(board, 0)
	
	// Assert
	assert.ErrorIs(t, err, ErrInsufficientArmy, "Should return ErrInsufficientArmy")
}

func TestMoveAction_ValidateNotOwned_ReturnsError(t *testing.T) {
	// Arrange
	board := NewBoard(5, 5)
	board.T[board.Idx(1, 1)] = Tile{Owner: 1, Army: 5, Type: TileNormal} // Owned by player 1
	
	action := &MoveAction{
		PlayerID: 0, // Player 0 trying to move player 1's tile
		FromX:    1, FromY: 1,
		ToX:      1, ToY: 2,
		MoveAll:  true,
	}
	
	// Act
	err := action.Validate(board, 0)
	
	// Assert
	assert.ErrorIs(t, err, ErrNotOwned, "Should return ErrNotOwned")
}

// Table-driven tests for comprehensive scenarios
func TestMoveAction_Validate_BoundaryConditions(t *testing.T) {
	board := NewBoard(3, 3)
	board.T[board.Idx(1, 1)] = Tile{Owner: 0, Army: 5, Type: TileNormal}
	
	tests := []struct {
		name      string
		action    MoveAction
		wantError error
	}{
		{
			name: "move_out_of_bounds_x",
			action: MoveAction{
				PlayerID: 0,
				FromX: 1, FromY: 1,
				ToX: 5, ToY: 1, // Out of bounds
				MoveAll: true,
			},
			wantError: ErrInvalidCoordinates,
		},
		{
			name: "move_out_of_bounds_y",
			action: MoveAction{
				PlayerID: 0,
				FromX: 1, FromY: 1,
				ToX: 1, ToY: 5, // Out of bounds
				MoveAll: true,
			},
			wantError: ErrInvalidCoordinates,
		},
		{
			name: "move_not_adjacent",
			action: MoveAction{
				PlayerID: 0,
				FromX: 1, FromY: 1,
				ToX: 1, ToY: 3, // Not adjacent
				MoveAll: true,
			},
			wantError: ErrNotAdjacent,
		},
		{
			name: "valid_move",
			action: MoveAction{
				PlayerID: 0,
				FromX: 1, FromY: 1,
				ToX: 1, ToY: 2, // Valid adjacent move
				MoveAll: true,
			},
			wantError: nil,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.action.Validate(board, tt.action.PlayerID)
			
			if tt.wantError != nil {
				assert.ErrorIs(t, err, tt.wantError)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}