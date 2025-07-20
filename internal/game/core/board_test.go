package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewBoard(t *testing.T) {
	tests := []struct {
		name   string
		width  int
		height int
	}{
		{"small board", 5, 5},
		{"rectangular board", 10, 20},
		{"large board", 100, 100},
		{"minimum board", 1, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			board := NewBoard(tt.width, tt.height)
			
			assert.Equal(t, tt.width, board.W)
			assert.Equal(t, tt.height, board.H)
			assert.Len(t, board.T, tt.width*tt.height)
			
			// Verify all tiles are initialized correctly
			for i, tile := range board.T {
				assert.Equal(t, NeutralID, tile.Owner, "tile %d should be neutral", i)
				assert.Equal(t, TileNormal, tile.Type, "tile %d should be normal type", i)
				assert.Equal(t, 0, tile.Army, "tile %d should have 0 army", i)
				assert.NotNil(t, tile.Visible, "tile %d should have visible map initialized", i)
			}
		})
	}
}

func TestBoard_Idx(t *testing.T) {
	board := NewBoard(5, 5)
	
	tests := []struct {
		x, y     int
		expected int
	}{
		{0, 0, 0},
		{4, 0, 4},
		{0, 1, 5},
		{2, 2, 12},
		{4, 4, 24},
	}
	
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			idx := board.Idx(tt.x, tt.y)
			assert.Equal(t, tt.expected, idx, "Idx(%d,%d) should be %d", tt.x, tt.y, tt.expected)
		})
	}
}

func TestBoard_XY(t *testing.T) {
	board := NewBoard(5, 5)
	
	tests := []struct {
		idx      int
		expectedX int
		expectedY int
	}{
		{0, 0, 0},
		{4, 4, 0},
		{5, 0, 1},
		{12, 2, 2},
		{24, 4, 4},
	}
	
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			x, y := board.XY(tt.idx)
			assert.Equal(t, tt.expectedX, x, "X coordinate for idx %d", tt.idx)
			assert.Equal(t, tt.expectedY, y, "Y coordinate for idx %d", tt.idx)
		})
	}
}

func TestBoard_InBounds(t *testing.T) {
	board := NewBoard(5, 5)
	
	tests := []struct {
		name     string
		x, y     int
		expected bool
	}{
		{"top-left corner", 0, 0, true},
		{"top-right corner", 4, 0, true},
		{"bottom-left corner", 0, 4, true},
		{"bottom-right corner", 4, 4, true},
		{"center", 2, 2, true},
		{"negative x", -1, 2, false},
		{"negative y", 2, -1, false},
		{"x too large", 5, 2, false},
		{"y too large", 2, 5, false},
		{"both negative", -1, -1, false},
		{"both too large", 10, 10, false},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := board.InBounds(tt.x, tt.y)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBoard_GetTile(t *testing.T) {
	board := NewBoard(5, 5)
	
	// Set up some test tiles
	board.T[0].Owner = 0
	board.T[0].Army = 10
	board.T[0].Type = TileGeneral
	
	board.T[12].Owner = 1  // (2,2)
	board.T[12].Army = 5
	board.T[12].Type = TileCity
	
	tests := []struct {
		name     string
		x, y     int
		expected *Tile
	}{
		{"valid general tile", 0, 0, &board.T[0]},
		{"valid city tile", 2, 2, &board.T[12]},
		{"out of bounds negative", -1, 0, nil},
		{"out of bounds large", 10, 10, nil},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tile := board.GetTile(tt.x, tt.y)
			if tt.expected == nil {
				assert.Nil(t, tile)
			} else {
				require.NotNil(t, tile)
				assert.Equal(t, tt.expected.Owner, tile.Owner)
				assert.Equal(t, tt.expected.Army, tile.Army)
				assert.Equal(t, tt.expected.Type, tile.Type)
			}
		})
	}
}

func TestBoard_Distance(t *testing.T) {
	board := NewBoard(10, 10)
	
	tests := []struct {
		name     string
		x1, y1   int
		x2, y2   int
		expected int
	}{
		{"same position", 5, 5, 5, 5, 0},
		{"horizontal distance", 0, 0, 5, 0, 5},
		{"vertical distance", 0, 0, 0, 5, 5},
		{"diagonal distance", 0, 0, 3, 4, 7},
		{"negative coords", 2, 2, -1, -1, 6},
		{"large distance", 0, 0, 9, 9, 18},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dist := board.Distance(tt.x1, tt.y1, tt.x2, tt.y2)
			assert.Equal(t, tt.expected, dist)
			
			// Distance should be symmetric
			distReverse := board.Distance(tt.x2, tt.y2, tt.x1, tt.y1)
			assert.Equal(t, dist, distReverse, "distance should be symmetric")
		})
	}
}

func TestTile_IsNeutral(t *testing.T) {
	tests := []struct {
		name     string
		owner    int
		expected bool
	}{
		{"neutral tile", NeutralID, true},
		{"player 0 tile", 0, false},
		{"player 1 tile", 1, false},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tile := Tile{Owner: tt.owner}
			assert.Equal(t, tt.expected, tile.IsNeutral())
		})
	}
}

func TestTile_TypeChecks(t *testing.T) {
	tests := []struct {
		name       string
		tileType   int
		isCity     bool
		isGeneral  bool
		isMountain bool
	}{
		{"normal tile", TileNormal, false, false, false},
		{"city tile", TileCity, true, false, false},
		{"general tile", TileGeneral, false, true, false},
		{"mountain tile", TileMountain, false, false, true},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tile := Tile{Type: tt.tileType}
			assert.Equal(t, tt.isCity, tile.IsCity())
			assert.Equal(t, tt.isGeneral, tile.IsGeneral())
			assert.Equal(t, tt.isMountain, tile.IsMountain())
		})
	}
}

func TestTile_IsEmpty(t *testing.T) {
	tests := []struct {
		name     string
		tile     Tile
		expected bool
	}{
		{"empty neutral tile", Tile{Owner: NeutralID, Type: TileNormal, Army: 0}, true},
		{"neutral with army", Tile{Owner: NeutralID, Type: TileNormal, Army: 5}, false},
		{"owned empty tile", Tile{Owner: 0, Type: TileNormal, Army: 0}, false},
		{"neutral city", Tile{Owner: NeutralID, Type: TileCity, Army: 0}, false},
		{"neutral general", Tile{Owner: NeutralID, Type: TileGeneral, Army: 0}, false},
		{"neutral mountain", Tile{Owner: NeutralID, Type: TileMountain, Army: 0}, false},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.tile.IsEmpty())
		})
	}
}

func TestBoard_EdgeCases(t *testing.T) {
	t.Run("1x1 board", func(t *testing.T) {
		board := NewBoard(1, 1)
		assert.Equal(t, 1, len(board.T))
		assert.True(t, board.InBounds(0, 0))
		assert.False(t, board.InBounds(1, 0))
		assert.False(t, board.InBounds(0, 1))
	})
	
	t.Run("very large board", func(t *testing.T) {
		board := NewBoard(1000, 1000)
		assert.Equal(t, 1000000, len(board.T))
		assert.True(t, board.InBounds(999, 999))
		assert.False(t, board.InBounds(1000, 1000))
	})
}

func TestBoard_ModifyTile(t *testing.T) {
	board := NewBoard(5, 5)
	
	// Get a tile and modify it
	tile := board.GetTile(2, 2)
	require.NotNil(t, tile)
	
	tile.Owner = 1
	tile.Army = 50
	tile.Type = TileCity
	
	// Verify modifications persist
	sameTile := board.GetTile(2, 2)
	assert.Equal(t, 1, sameTile.Owner)
	assert.Equal(t, 50, sameTile.Army)
	assert.Equal(t, TileCity, sameTile.Type)
}

func TestBoard_VisibilityMap(t *testing.T) {
	board := NewBoard(3, 3)
	
	// Set visibility for some tiles
	tile := board.GetTile(1, 1)
	tile.Visible[0] = true
	tile.Visible[1] = false
	
	// Verify visibility settings
	assert.True(t, tile.Visible[0])
	assert.False(t, tile.Visible[1])
	assert.False(t, tile.Visible[2]) // unset defaults to false
}

func TestTileBitfieldVisibility(t *testing.T) {
	t.Run("BasicVisibilityOperations", func(t *testing.T) {
		tile := &Tile{VisibleBitfield: 0, Visible: make(map[int]bool)}
		
		// Test setting visibility
		tile.SetVisible(0, true)
		assert.True(t, tile.IsVisibleTo(0))
		assert.False(t, tile.IsVisibleTo(1))
		
		// Test setting multiple players
		tile.SetVisible(3, true)
		tile.SetVisible(7, true)
		assert.True(t, tile.IsVisibleTo(0))
		assert.True(t, tile.IsVisibleTo(3))
		assert.True(t, tile.IsVisibleTo(7))
		assert.False(t, tile.IsVisibleTo(1))
		assert.False(t, tile.IsVisibleTo(2))
		
		// Test clearing visibility
		tile.SetVisible(0, false)
		assert.False(t, tile.IsVisibleTo(0))
		assert.True(t, tile.IsVisibleTo(3))
		assert.True(t, tile.IsVisibleTo(7))
	})

	t.Run("BoundaryConditions", func(t *testing.T) {
		tile := &Tile{VisibleBitfield: 0}
		
		// Test invalid player IDs
		tile.SetVisible(-1, true)
		assert.False(t, tile.IsVisibleTo(-1))
		
		tile.SetVisible(32, true)
		assert.False(t, tile.IsVisibleTo(32))
		
		// Test max valid player ID (31)
		tile.SetVisible(31, true)
		assert.True(t, tile.IsVisibleTo(31))
	})

	t.Run("AllPlayersVisible", func(t *testing.T) {
		tile := &Tile{VisibleBitfield: 0}
		
		// Set all 32 players as visible
		for i := 0; i < 32; i++ {
			tile.SetVisible(i, true)
		}
		
		// Check all are visible
		for i := 0; i < 32; i++ {
			assert.True(t, tile.IsVisibleTo(i))
		}
		
		// Clear one in the middle
		tile.SetVisible(15, false)
		assert.False(t, tile.IsVisibleTo(15))
		assert.True(t, tile.IsVisibleTo(14))
		assert.True(t, tile.IsVisibleTo(16))
	})

	t.Run("CompatibilityLayer", func(t *testing.T) {
		tile := &Tile{
			VisibleBitfield: 0,
			Visible: make(map[int]bool),
		}
		
		// Test sync from map to bitfield
		tile.Visible[0] = true
		tile.Visible[5] = true
		tile.Visible[10] = true
		tile.SyncVisibilityFromMap()
		
		assert.True(t, tile.IsVisibleTo(0))
		assert.True(t, tile.IsVisibleTo(5))
		assert.True(t, tile.IsVisibleTo(10))
		assert.False(t, tile.IsVisibleTo(1))
		
		// Test sync from bitfield to map
		tile.SetVisible(15, true)
		tile.SetVisible(20, true)
		tile.SyncVisibilityToMap()
		
		assert.True(t, tile.Visible[0])
		assert.True(t, tile.Visible[5])
		assert.True(t, tile.Visible[10])
		assert.True(t, tile.Visible[15])
		assert.True(t, tile.Visible[20])
		assert.False(t, tile.Visible[1])
		
		// Verify map doesn't have entries for false values
		_, exists := tile.Visible[1]
		assert.False(t, exists)
	})

	t.Run("BitfieldDirectManipulation", func(t *testing.T) {
		tile := &Tile{VisibleBitfield: 0}
		
		// Directly set some bits
		tile.VisibleBitfield = 0b10101010
		
		assert.False(t, tile.IsVisibleTo(0))
		assert.True(t, tile.IsVisibleTo(1))
		assert.False(t, tile.IsVisibleTo(2))
		assert.True(t, tile.IsVisibleTo(3))
		assert.False(t, tile.IsVisibleTo(4))
		assert.True(t, tile.IsVisibleTo(5))
		assert.False(t, tile.IsVisibleTo(6))
		assert.True(t, tile.IsVisibleTo(7))
	})
}