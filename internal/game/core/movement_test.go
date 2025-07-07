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