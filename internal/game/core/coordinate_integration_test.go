package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestCoordinateIntegration tests the new Coordinate methods integrated into Board, MoveAction, and CaptureDetails
func TestCoordinateIntegration(t *testing.T) {
	// Test Board coordinate methods
	t.Run("Board Coordinate Methods", func(t *testing.T) {
		board := NewBoard(10, 10)
		coord := NewCoordinate(5, 5)
		
		// Test InBoundsCoord
		assert.True(t, board.InBoundsCoord(coord))
		assert.False(t, board.InBoundsCoord(NewCoordinate(-1, 5)))
		assert.False(t, board.InBoundsCoord(NewCoordinate(10, 5)))
		
		// Test GetTileCoord
		tile := board.GetTileCoord(coord)
		assert.NotNil(t, tile)
		assert.Equal(t, NeutralID, tile.Owner)
		
		// Test SetTile
		newTile := Tile{Owner: 1, Army: 10, Type: TileNormal}
		board.SetTile(coord, newTile)
		
		retrievedTile := board.GetTileCoord(coord)
		assert.Equal(t, 1, retrievedTile.Owner)
		assert.Equal(t, 10, retrievedTile.Army)
		
		// Test IdxCoord
		idx := board.IdxCoord(coord)
		assert.Equal(t, 55, idx) // 5 * 10 + 5
		
		// Test SetTile with out of bounds coordinate
		board.SetTile(NewCoordinate(-1, -1), newTile) // Should not panic
	})

	// Test MoveAction coordinate methods
	t.Run("MoveAction Coordinate Methods", func(t *testing.T) {
		// Test with legacy fields
		action1 := &MoveAction{
			PlayerID: 1,
			FromX: 3, FromY: 4,
			ToX: 3, ToY: 5,
			MoveAll: true,
		}
		
		from1 := action1.GetFrom()
		to1 := action1.GetTo()
		assert.Equal(t, Coordinate{X: 3, Y: 4}, from1)
		assert.Equal(t, Coordinate{X: 3, Y: 5}, to1)
		
		// Test with new coordinate fields
		action2 := &MoveAction{
			PlayerID: 1,
			From: NewCoordinate(5, 6),
			To: NewCoordinate(5, 7),
			// Legacy fields should be ignored when From/To are set
			FromX: 0, FromY: 0,
			ToX: 0, ToY: 0,
			MoveAll: false,
		}
		
		from2 := action2.GetFrom()
		to2 := action2.GetTo()
		assert.Equal(t, Coordinate{X: 5, Y: 6}, from2)
		assert.Equal(t, Coordinate{X: 5, Y: 7}, to2)
		
		// Test mixed usage (not recommended but should work)
		action3 := &MoveAction{
			PlayerID: 1,
			From: NewCoordinate(2, 3),
			// To not set, should fall back to ToX, ToY
			ToX: 2, ToY: 4,
			MoveAll: true,
		}
		
		from3 := action3.GetFrom()
		to3 := action3.GetTo()
		assert.Equal(t, Coordinate{X: 2, Y: 3}, from3)
		assert.Equal(t, Coordinate{X: 2, Y: 4}, to3)
	})
	
	// Test CaptureDetails with Location field
	t.Run("CaptureDetails Location Field", func(t *testing.T) {
		board := NewBoard(10, 10)
		
		// Set up tiles for capture
		board.T[board.Idx(3, 3)] = Tile{Owner: 1, Army: 10, Type: TileNormal}
		board.T[board.Idx(3, 4)] = Tile{Owner: 2, Army: 5, Type: TileCity}
		
		action := &MoveAction{
			PlayerID: 1,
			FromX: 3, FromY: 3,
			ToX: 3, ToY: 4,
			MoveAll: true,
		}
		
		capture, err := ApplyMoveAction(board, action, nil)
		assert.NoError(t, err)
		assert.NotNil(t, capture)
		
		// Verify capture details
		assert.Equal(t, 3, capture.X)
		assert.Equal(t, 4, capture.Y)
		assert.Equal(t, Coordinate{X: 3, Y: 4}, capture.Location)
		assert.Equal(t, TileCity, capture.TileType)
		assert.Equal(t, 1, capture.CapturingPlayerID)
		assert.Equal(t, 2, capture.PreviousOwnerID)
		assert.Equal(t, 5, capture.PreviousArmyCount)
	})
}