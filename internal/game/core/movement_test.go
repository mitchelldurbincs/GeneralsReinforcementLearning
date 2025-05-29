package core

import (
	"testing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestApplyMoveAction_SuccessfulCapture_UpdatesBoardCorrectly(t *testing.T) {
	// Arrange
	board := NewBoard(3, 3)
	// Attacker: Player 0 with 5 armies
	board.T[board.Idx(0, 0)] = Tile{Owner: 0, Army: 5, Type: TileNormal}
	// Defender: Player 1 with 2 armies  
	board.T[board.Idx(0, 1)] = Tile{Owner: 1, Army: 2, Type: TileNormal}
	
	action := &MoveAction{
		PlayerID: 0,
		FromX: 0, FromY: 0,
		ToX: 0, ToY: 1,
		MoveAll: true, // Move 4 armies (leave 1 behind)
	}
	
	// Act
	captured, err := ApplyMoveAction(board, action)
	
	// Assert
	require.NoError(t, err, "Valid move should not error")
	assert.True(t, captured, "Should have captured the tile")
	
	// Check source tile
	sourceTile := board.T[board.Idx(0, 0)]
	assert.Equal(t, 0, sourceTile.Owner, "Source should still be owned by player 0")
	assert.Equal(t, 1, sourceTile.Army, "Source should have 1 army remaining")
	
	// Check destination tile
	destTile := board.T[board.Idx(0, 1)]
	assert.Equal(t, 0, destTile.Owner, "Destination should now be owned by player 0")
	assert.Equal(t, 2, destTile.Army, "Destination should have 2 armies (4 attacking - 2 defending)")
}

func TestApplyMoveAction_FailedAttack_DefenderKeepsTile(t *testing.T) {
	// Arrange
	board := NewBoard(3, 3)
	// Weak attacker: Player 0 with 2 armies
	board.T[board.Idx(0, 0)] = Tile{Owner: 0, Army: 2, Type: TileNormal}
	// Strong defender: Player 1 with 5 armies
	board.T[board.Idx(0, 1)] = Tile{Owner: 1, Army: 5, Type: TileNormal}
	
	action := &MoveAction{
		PlayerID: 0,
		FromX: 0, FromY: 0,
		ToX: 0, ToY: 1,
		MoveAll: true, // Move 1 army (leave 1 behind)
	}
	
	// Act
	captured, err := ApplyMoveAction(board, action)
	
	// Assert
	require.NoError(t, err, "Valid move should not error")
	assert.False(t, captured, "Should not have captured the tile")
	
	// Check destination tile
	destTile := board.T[board.Idx(0, 1)]
	assert.Equal(t, 1, destTile.Owner, "Destination should still be owned by player 1")
	assert.Equal(t, 4, destTile.Army, "Destination should have 4 armies (5 - 1 attacking)")
}

func TestApplyMoveAction_MoveToOwnTile_CombinesArmies(t *testing.T) {
	// Arrange
	board := NewBoard(3, 3)
	board.T[board.Idx(0, 0)] = Tile{Owner: 0, Army: 5, Type: TileNormal}
	board.T[board.Idx(0, 1)] = Tile{Owner: 0, Army: 3, Type: TileNormal} // Same owner
	
	action := &MoveAction{
		PlayerID: 0,
		FromX: 0, FromY: 0,
		ToX: 0, ToY: 1,
		MoveAll: false, // Move half (2 armies)
	}
	
	// Act
	captured, err := ApplyMoveAction(board, action)
	
	// Assert
	require.NoError(t, err, "Valid move should not error")
	assert.False(t, captured, "Moving to own tile is not a capture")
	
	// Check source tile
	sourceTile := board.T[board.Idx(0, 0)]
	assert.Equal(t, 3, sourceTile.Army, "Source should have 3 armies remaining (5 - 2)")
	
	// Check destination tile
	destTile := board.T[board.Idx(0, 1)]
	assert.Equal(t, 5, destTile.Army, "Destination should have 5 armies (3 + 2)")
}

// Golden Test: Complete battle scenario with known outcome
func TestApplyMoveAction_GoldenScenario_KnownOutcome(t *testing.T) {
	// This test uses a specific scenario that should always produce the same result
	// Useful for regression testing and ensuring determinism
	
	// Arrange - Set up a specific battle scenario
	board := NewBoard(5, 5)
	
	// Player 0 positions
	board.T[board.Idx(1, 1)] = Tile{Owner: 0, Army: 10, Type: TileGeneral}
	board.T[board.Idx(2, 1)] = Tile{Owner: 0, Army: 8, Type: TileNormal}
	board.T[board.Idx(1, 2)] = Tile{Owner: 0, Army: 6, Type: TileNormal}
	
	// Player 1 positions  
	board.T[board.Idx(3, 1)] = Tile{Owner: 1, Army: 5, Type: TileNormal}
	board.T[board.Idx(3, 2)] = Tile{Owner: 1, Army: 7, Type: TileCity}
	
	// Neutral city
	board.T[board.Idx(2, 3)] = Tile{Owner: NeutralID, Army: 40, Type: TileCity}
	
	actions := []*MoveAction{
		{PlayerID: 0, FromX: 2, FromY: 1, ToX: 3, ToY: 1, MoveAll: true}, // Attack player 1
		{PlayerID: 0, FromX: 1, FromY: 2, ToX: 2, ToY: 2, MoveAll: false}, // Move to empty
	}
	
	// Act & Assert each move
	captured1, err1 := ApplyMoveAction(board, actions[0])
	require.NoError(t, err1)
	assert.True(t, captured1, "First attack should succeed")
	
	captured2, err2 := ApplyMoveAction(board, actions[1])
	require.NoError(t, err2)
	assert.False(t, captured2, "Moving to empty is not a capture")
	
	// Verify final board state
	assert.Equal(t, 0, board.T[board.Idx(3, 1)].Owner, "Tile should be captured by player 0")
	assert.Equal(t, 2, board.T[board.Idx(3, 1)].Army, "Should have 2 armies (7 attacking - 5 defending)")
	assert.Equal(t, 0, board.T[board.Idx(2, 2)].Owner, "Should be owned by player 0")
	assert.Equal(t, 3, board.T[board.Idx(2, 2)].Army, "Should have 3 armies (half of 6)")
}