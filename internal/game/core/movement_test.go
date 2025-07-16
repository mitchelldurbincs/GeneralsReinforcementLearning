package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCaptureGeneralAndEliminationOrder(t *testing.T) {
	board := NewBoard(3, 3)

	p0AttackerIdx := board.Idx(0, 0)
	board.T[p0AttackerIdx].Owner = 0
	board.T[p0AttackerIdx].Army = 10
	board.T[p0AttackerIdx].Type = TileNormal

	p1GeneralIdx := board.Idx(1, 0)
	board.T[p1GeneralIdx].Owner = 1
	board.T[p1GeneralIdx].Army = 1
	board.T[p1GeneralIdx].Type = TileGeneral

	p1CityIdx := board.Idx(1, 1)
	board.T[p1CityIdx].Owner = 1
	board.T[p1CityIdx].Army = 5
	board.T[p1CityIdx].Type = TileCity

	p1LandIdx := board.Idx(2, 1)
	board.T[p1LandIdx].Owner = 1
	board.T[p1LandIdx].Army = 3
	board.T[p1LandIdx].Type = TileNormal

	moveAction := MoveAction{
		PlayerID: 0,
		FromX:    0, FromY: 0,
		ToX:      1, ToY: 0,
		MoveAll:  true,
	}

	captureDetails, err := ApplyMoveAction(board, &moveAction, nil)

	require.NoError(t, err, "ApplyMoveAction returned an unexpected error")
	require.NotNil(t, captureDetails, "ApplyMoveAction should have resulted in a capture, but captureDetails is nil")

	assert.Equal(t, 1, captureDetails.X, "CaptureDetails: X coordinate mismatch")
	assert.Equal(t, 0, captureDetails.Y, "CaptureDetails: Y coordinate mismatch")
	assert.Equal(t, TileGeneral, captureDetails.TileType, "CaptureDetails: TileType mismatch")
	assert.Equal(t, 0, captureDetails.CapturingPlayerID, "CaptureDetails: CapturingPlayerID mismatch")
	assert.Equal(t, 1, captureDetails.PreviousOwnerID, "CaptureDetails: PreviousOwnerID mismatch")
	assert.Equal(t, 1, captureDetails.PreviousArmyCount, "CaptureDetails: PreviousArmyCount mismatch")

	assert.Equal(t, 0, board.T[p0AttackerIdx].Owner, "Attacker tile (0,0) Owner mismatch")
	assert.Equal(t, 1, board.T[p0AttackerIdx].Army, "Attacker tile (0,0) Army mismatch")

	expectedArmyOnCapturedTile := (10 - 1) - 1
	assert.Equal(t, 0, board.T[p1GeneralIdx].Owner, "Captured General tile (1,0) Owner mismatch")
	assert.Equal(t, expectedArmyOnCapturedTile, board.T[p1GeneralIdx].Army, "Captured General tile (1,0) Army mismatch")
	assert.Equal(t, TileGeneral, board.T[p1GeneralIdx].Type, "Captured General tile (1,0) Type should remain General")

	eliminationOrders := ProcessCaptures([]CaptureDetails{*captureDetails})

	require.Len(t, eliminationOrders, 1, "ProcessCaptures: expected 1 elimination order")

	order := eliminationOrders[0]
	assert.Equal(t, 1, order.EliminatedPlayerID, "EliminationOrder: EliminatedPlayerID mismatch")
	assert.Equal(t, 0, order.NewOwnerID, "EliminationOrder: NewOwnerID mismatch")

	t.Log("TestCaptureGeneralAndEliminationOrder PASSED core logic checks with testify.")
}

func TestApplyMoveAction_BasicMovement(t *testing.T) {
	tests := []struct {
		name           string
		setupBoard     func() *Board
		action         MoveAction
		expectedFromArmy int
		expectedToArmy   int
		expectedToOwner  int
		expectCapture    bool
	}{
		{
			name: "move all to own tile",
			setupBoard: func() *Board {
				b := NewBoard(3, 3)
				b.T[0].Owner = 0
				b.T[0].Army = 10
				b.T[1].Owner = 0
				b.T[1].Army = 5
				return b
			},
			action: MoveAction{
				PlayerID: 0,
				FromX: 0, FromY: 0,
				ToX: 1, ToY: 0,
				MoveAll: true,
			},
			expectedFromArmy: 1,
			expectedToArmy:   14, // 5 + 9
			expectedToOwner:  0,
			expectCapture:    false,
		},
		{
			name: "move half to own tile",
			setupBoard: func() *Board {
				b := NewBoard(3, 3)
				b.T[0].Owner = 0
				b.T[0].Army = 10
				b.T[1].Owner = 0
				b.T[1].Army = 5
				return b
			},
			action: MoveAction{
				PlayerID: 0,
				FromX: 0, FromY: 0,
				ToX: 1, ToY: 0,
				MoveAll: false,
			},
			expectedFromArmy: 5,
			expectedToArmy:   10, // 5 + 5
			expectedToOwner:  0,
			expectCapture:    false,
		},
		{
			name: "capture neutral tile",
			setupBoard: func() *Board {
				b := NewBoard(3, 3)
				b.T[0].Owner = 0
				b.T[0].Army = 10
				b.T[1].Owner = NeutralID
				b.T[1].Army = 3
				return b
			},
			action: MoveAction{
				PlayerID: 0,
				FromX: 0, FromY: 0,
				ToX: 1, ToY: 0,
				MoveAll: true,
			},
			expectedFromArmy: 1,
			expectedToArmy:   6, // 9 - 3
			expectedToOwner:  0,
			expectCapture:    true,
		},
		{
			name: "failed attack - defender stronger",
			setupBoard: func() *Board {
				b := NewBoard(3, 3)
				b.T[0].Owner = 0
				b.T[0].Army = 5
				b.T[1].Owner = 1
				b.T[1].Army = 10
				return b
			},
			action: MoveAction{
				PlayerID: 0,
				FromX: 0, FromY: 0,
				ToX: 1, ToY: 0,
				MoveAll: true,
			},
			expectedFromArmy: 1,
			expectedToArmy:   6, // 10 - 4
			expectedToOwner:  1,
			expectCapture:    false,
		},
		{
			name: "equal armies - defender wins",
			setupBoard: func() *Board {
				b := NewBoard(3, 3)
				b.T[0].Owner = 0
				b.T[0].Army = 6
				b.T[1].Owner = 1
				b.T[1].Army = 5
				return b
			},
			action: MoveAction{
				PlayerID: 0,
				FromX: 0, FromY: 0,
				ToX: 1, ToY: 0,
				MoveAll: true,
			},
			expectedFromArmy: 1,
			expectedToArmy:   0, // 5 - 5
			expectedToOwner:  1,
			expectCapture:    false,
		},
		{
			name: "move half with small army",
			setupBoard: func() *Board {
				b := NewBoard(3, 3)
				b.T[0].Owner = 0
				b.T[0].Army = 3
				b.T[1].Owner = 0
				b.T[1].Army = 0
				return b
			},
			action: MoveAction{
				PlayerID: 0,
				FromX: 0, FromY: 0,
				ToX: 1, ToY: 0,
				MoveAll: false,
			},
			expectedFromArmy: 2,
			expectedToArmy:   1, // half of 3 is 1
			expectedToOwner:  0,
			expectCapture:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			board := tt.setupBoard()
			changedTiles := make(map[int]struct{})
			
			capture, err := ApplyMoveAction(board, &tt.action, changedTiles)
			require.NoError(t, err)
			
			fromIdx := board.Idx(tt.action.FromX, tt.action.FromY)
			toIdx := board.Idx(tt.action.ToX, tt.action.ToY)
			
			assert.Equal(t, tt.expectedFromArmy, board.T[fromIdx].Army, "from tile army")
			assert.Equal(t, tt.expectedToArmy, board.T[toIdx].Army, "to tile army")
			assert.Equal(t, tt.expectedToOwner, board.T[toIdx].Owner, "to tile owner")
			
			if tt.expectCapture {
				assert.NotNil(t, capture, "expected capture details")
			} else {
				assert.Nil(t, capture, "expected no capture")
			}
			
			// Verify changed tiles tracking
			assert.Contains(t, changedTiles, fromIdx)
			assert.Contains(t, changedTiles, toIdx)
		})
	}
}

func TestApplyMoveAction_CaptureDetails(t *testing.T) {
	board := NewBoard(5, 5)
	
	// Set up a city owned by player 1
	cityIdx := board.Idx(2, 2)
	board.T[cityIdx].Owner = 1
	board.T[cityIdx].Army = 40
	board.T[cityIdx].Type = TileCity
	
	// Player 0 attacks with overwhelming force
	attackerIdx := board.Idx(2, 1)
	board.T[attackerIdx].Owner = 0
	board.T[attackerIdx].Army = 50
	
	action := MoveAction{
		PlayerID: 0,
		FromX: 2, FromY: 1,
		ToX: 2, ToY: 2,
		MoveAll: true,
	}
	
	capture, err := ApplyMoveAction(board, &action, nil)
	require.NoError(t, err)
	require.NotNil(t, capture)
	
	assert.Equal(t, 2, capture.X)
	assert.Equal(t, 2, capture.Y)
	assert.Equal(t, TileCity, capture.TileType)
	assert.Equal(t, 0, capture.CapturingPlayerID)
	assert.Equal(t, 1, capture.PreviousOwnerID)
	assert.Equal(t, 40, capture.PreviousArmyCount)
	
	// Verify the city now belongs to player 0
	assert.Equal(t, 0, board.T[cityIdx].Owner)
	assert.Equal(t, 9, board.T[cityIdx].Army) // 49 - 40
	assert.Equal(t, TileCity, board.T[cityIdx].Type) // Type doesn't change
}

func TestApplyMoveAction_InvalidMoves(t *testing.T) {
	board := NewBoard(5, 5)
	
	// Set up player tile
	board.T[0].Owner = 0
	board.T[0].Army = 10
	
	// Set up mountain
	mountainIdx := board.Idx(1, 0)
	board.T[mountainIdx].Type = TileMountain
	
	tests := []struct {
		name   string
		action MoveAction
	}{
		{
			name: "move to mountain",
			action: MoveAction{
				PlayerID: 0,
				FromX: 0, FromY: 0,
				ToX: 1, ToY: 0,
				MoveAll: true,
			},
		},
		{
			name: "move from unowned tile",
			action: MoveAction{
				PlayerID: 0,
				FromX: 3, FromY: 3,
				ToX: 3, ToY: 4,
				MoveAll: true,
			},
		},
		{
			name: "move with insufficient army",
			action: MoveAction{
				PlayerID: 0,
				FromX: 0, FromY: 0,
				ToX: 1, ToY: 0,
				MoveAll: true,
			},
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Special setup for insufficient army test
			if tt.name == "move with insufficient army" {
				board.T[0].Army = 1
				board.T[1].Type = TileNormal // Not a mountain
			}
			
			_, err := ApplyMoveAction(board, &tt.action, nil)
			assert.Error(t, err, "expected error for invalid move")
		})
	}
}

func TestProcessCaptures(t *testing.T) {
	tests := []struct {
		name              string
		captures          []CaptureDetails
		expectedEliminations []PlayerEliminationOrder
	}{
		{
			name: "single general capture",
			captures: []CaptureDetails{
				{
					X: 1, Y: 1,
					TileType:          TileGeneral,
					CapturingPlayerID: 0,
					PreviousOwnerID:   1,
					PreviousArmyCount: 1,
				},
			},
			expectedEliminations: []PlayerEliminationOrder{
				{
					EliminatedPlayerID: 1,
					NewOwnerID:         0,
				},
			},
		},
		{
			name: "multiple captures including general",
			captures: []CaptureDetails{
				{
					X: 1, Y: 1,
					TileType:          TileCity,
					CapturingPlayerID: 0,
					PreviousOwnerID:   1,
					PreviousArmyCount: 40,
				},
				{
					X: 2, Y: 2,
					TileType:          TileGeneral,
					CapturingPlayerID: 0,
					PreviousOwnerID:   2,
					PreviousArmyCount: 1,
				},
				{
					X: 3, Y: 3,
					TileType:          TileNormal,
					CapturingPlayerID: 1,
					PreviousOwnerID:   2,
					PreviousArmyCount: 5,
				},
			},
			expectedEliminations: []PlayerEliminationOrder{
				{
					EliminatedPlayerID: 2,
					NewOwnerID:         0,
				},
			},
		},
		{
			name: "no general captures",
			captures: []CaptureDetails{
				{
					X: 1, Y: 1,
					TileType:          TileCity,
					CapturingPlayerID: 0,
					PreviousOwnerID:   1,
					PreviousArmyCount: 40,
				},
				{
					X: 3, Y: 3,
					TileType:          TileNormal,
					CapturingPlayerID: 1,
					PreviousOwnerID:   2,
					PreviousArmyCount: 5,
				},
			},
			expectedEliminations: []PlayerEliminationOrder{},
		},
		{
			name: "neutral general capture (no elimination)",
			captures: []CaptureDetails{
				{
					X: 1, Y: 1,
					TileType:          TileGeneral,
					CapturingPlayerID: 0,
					PreviousOwnerID:   NeutralID,
					PreviousArmyCount: 1,
				},
			},
			expectedEliminations: []PlayerEliminationOrder{},
		},
		{
			name: "duplicate general captures (same player)",
			captures: []CaptureDetails{
				{
					X: 1, Y: 1,
					TileType:          TileGeneral,
					CapturingPlayerID: 0,
					PreviousOwnerID:   1,
					PreviousArmyCount: 1,
				},
				{
					X: 2, Y: 2,
					TileType:          TileGeneral,
					CapturingPlayerID: 2,
					PreviousOwnerID:   1,
					PreviousArmyCount: 1,
				},
			},
			expectedEliminations: []PlayerEliminationOrder{
				{
					EliminatedPlayerID: 1,
					NewOwnerID:         0,
				},
			},
		},
		{
			name:                 "empty captures list",
			captures:             []CaptureDetails{},
			expectedEliminations: []PlayerEliminationOrder{},
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eliminations := ProcessCaptures(tt.captures)
			
			if len(tt.expectedEliminations) == 0 {
				assert.Empty(t, eliminations)
			} else {
				assert.Equal(t, tt.expectedEliminations, eliminations)
			}
		})
	}
}

func TestApplyMoveAction_EdgeCases(t *testing.T) {
	t.Run("nil changed tiles map", func(t *testing.T) {
		board := NewBoard(3, 3)
		board.T[0].Owner = 0
		board.T[0].Army = 10
		
		action := MoveAction{
			PlayerID: 0,
			FromX: 0, FromY: 0,
			ToX: 1, ToY: 0,
			MoveAll: true,
		}
		
		// Should not panic with nil map
		_, err := ApplyMoveAction(board, &action, nil)
		assert.NoError(t, err)
	})
	
	t.Run("move minimum army with half", func(t *testing.T) {
		board := NewBoard(3, 3)
		board.T[0].Owner = 0
		board.T[0].Army = 2
		board.T[1].Owner = 0
		board.T[1].Army = 0
		
		action := MoveAction{
			PlayerID: 0,
			FromX: 0, FromY: 0,
			ToX: 1, ToY: 0,
			MoveAll: false,
		}
		
		_, err := ApplyMoveAction(board, &action, nil)
		assert.NoError(t, err)
		assert.Equal(t, 1, board.T[0].Army)
		assert.Equal(t, 1, board.T[1].Army)
	})
}