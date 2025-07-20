package game

import (
	"math/rand"
	"testing"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// createTestEngineForActionMask creates an engine with a custom board for testing action masks
func createTestEngineForActionMask(width, height, players int) *Engine {
	// Create a simple board without map generation
	board := core.NewBoard(width, height)
	
	// Create game state
	gs := &GameState{
		Board:                  board,
		Players:                make([]Player, players),
		Turn:                   0,
		FogOfWarEnabled:        true,
		ChangedTiles:           make(map[int]struct{}),
		VisibilityChangedTiles: make(map[int]struct{}),
	}
	
	// Initialize players
	for i := 0; i < players; i++ {
		gs.Players[i] = Player{
			ID:         i,
			Alive:      true,
			ArmyCount:  0,
			GeneralIdx: -1,
			OwnedTiles: []int{},
		}
	}
	
	// Create engine
	engine := &Engine{
		gs:                  gs,
		rng:                 rand.New(rand.NewSource(42)),
		gameOver:            false,
		logger:              zerolog.Nop(),
		tempTileOwnership:   make(map[int]int),
		tempAffectedPlayers: make(map[int]struct{}),
	}
	
	return engine
}

func TestGetLegalActionMask_BasicScenario(t *testing.T) {
	// Create a simple 3x3 board for testing
	engine := createTestEngineForActionMask(3, 3, 2)
	require.NotNil(t, engine)

	// Manually set up a test scenario
	// Player 0 owns tile (1,1) with 5 armies
	centerIdx := engine.gs.Board.Idx(1, 1)
	engine.gs.Board.T[centerIdx].Owner = 0
	engine.gs.Board.T[centerIdx].Army = 5
	engine.gs.Players[0].OwnedTiles = []int{centerIdx}
	engine.gs.Players[0].Alive = true

	// Get action mask for player 0
	mask := engine.GetLegalActionMask(0)
	
	// Total actions should be 3*3*4 = 36
	assert.Len(t, mask, 36)
	
	// Check that only moves from (1,1) are legal
	// From (1,1), we can move in all 4 directions
	centerActionBase := (1*3 + 1) * 4
	
	// Up (1,1) -> (1,0)
	assert.True(t, mask[centerActionBase+0], "Should be able to move up")
	// Right (1,1) -> (2,1)
	assert.True(t, mask[centerActionBase+1], "Should be able to move right")
	// Down (1,1) -> (1,2)
	assert.True(t, mask[centerActionBase+2], "Should be able to move down")
	// Left (1,1) -> (0,1)
	assert.True(t, mask[centerActionBase+3], "Should be able to move left")
	
	// All other actions should be false
	legalCount := 0
	for i, legal := range mask {
		if legal {
			legalCount++
			// Verify this is one of the expected moves
			assert.True(t, i >= centerActionBase && i < centerActionBase+4,
				"Unexpected legal action at index %d", i)
		}
	}
	assert.Equal(t, 4, legalCount, "Should have exactly 4 legal moves")
}

func TestGetLegalActionMask_EdgeTiles(t *testing.T) {
	// Test movement from edge tiles (fewer legal moves)
	engine := createTestEngineForActionMask(3, 3, 1)
	require.NotNil(t, engine)

	// Player 0 owns corner tile (0,0) with 3 armies
	cornerIdx := engine.gs.Board.Idx(0, 0)
	engine.gs.Board.T[cornerIdx].Owner = 0
	engine.gs.Board.T[cornerIdx].Army = 3
	engine.gs.Players[0].OwnedTiles = []int{cornerIdx}
	engine.gs.Players[0].Alive = true

	mask := engine.GetLegalActionMask(0)
	
	// From corner (0,0), can only move right or down
	cornerActionBase := (0*3 + 0) * 4
	
	// Up would be out of bounds
	assert.False(t, mask[cornerActionBase+0], "Cannot move up from top edge")
	// Right (0,0) -> (1,0) is valid
	assert.True(t, mask[cornerActionBase+1], "Should be able to move right")
	// Down (0,0) -> (0,1) is valid
	assert.True(t, mask[cornerActionBase+2], "Should be able to move down")
	// Left would be out of bounds
	assert.False(t, mask[cornerActionBase+3], "Cannot move left from left edge")
}

func TestGetLegalActionMask_InsufficientArmy(t *testing.T) {
	// Test that tiles with 1 or fewer armies cannot move
	engine := createTestEngineForActionMask(3, 3, 1)
	require.NotNil(t, engine)

	// Player 0 owns two tiles
	tile1Idx := engine.gs.Board.Idx(0, 0)
	tile2Idx := engine.gs.Board.Idx(1, 1)
	
	// Tile 1 has only 1 army (cannot move)
	engine.gs.Board.T[tile1Idx].Owner = 0
	engine.gs.Board.T[tile1Idx].Army = 1
	
	// Tile 2 has 2 armies (can move)
	engine.gs.Board.T[tile2Idx].Owner = 0
	engine.gs.Board.T[tile2Idx].Army = 2
	
	engine.gs.Players[0].OwnedTiles = []int{tile1Idx, tile2Idx}
	engine.gs.Players[0].Alive = true

	mask := engine.GetLegalActionMask(0)
	
	// No moves from tile1 (insufficient army)
	tile1ActionBase := (0*3 + 0) * 4
	for dir := 0; dir < 4; dir++ {
		assert.False(t, mask[tile1ActionBase+dir], 
			"Should not be able to move from tile with 1 army")
	}
	
	// Should have moves from tile2
	tile2ActionBase := (1*3 + 1) * 4
	hasLegalMove := false
	for dir := 0; dir < 4; dir++ {
		if mask[tile2ActionBase+dir] {
			hasLegalMove = true
			break
		}
	}
	assert.True(t, hasLegalMove, "Should have at least one legal move from tile with 2 armies")
}

func TestGetLegalActionMask_Mountains(t *testing.T) {
	// Test that moves to mountain tiles are illegal
	engine := createTestEngineForActionMask(3, 3, 1)
	require.NotNil(t, engine)

	// Player 0 owns center tile
	centerIdx := engine.gs.Board.Idx(1, 1)
	engine.gs.Board.T[centerIdx].Owner = 0
	engine.gs.Board.T[centerIdx].Army = 5
	engine.gs.Players[0].OwnedTiles = []int{centerIdx}
	engine.gs.Players[0].Alive = true
	
	// Make all surrounding tiles mountains
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			if dx == 0 && dy == 0 {
				continue // Skip center
			}
			x, y := 1+dx, 1+dy
			if x >= 0 && x < 3 && y >= 0 && y < 3 {
				idx := engine.gs.Board.Idx(x, y)
				engine.gs.Board.T[idx].Type = core.TileMountain
			}
		}
	}

	mask := engine.GetLegalActionMask(0)
	
	// All moves should be illegal (mountains)
	centerActionBase := (1*3 + 1) * 4
	for dir := 0; dir < 4; dir++ {
		assert.False(t, mask[centerActionBase+dir], 
			"Should not be able to move to mountain tile")
	}
}

func TestGetLegalActionMask_DeadPlayer(t *testing.T) {
	// Test that dead players have no legal moves
	engine := createTestEngineForActionMask(3, 3, 2)
	require.NotNil(t, engine)

	// Player 0 is dead but still owns a tile
	tileIdx := engine.gs.Board.Idx(1, 1)
	engine.gs.Board.T[tileIdx].Owner = 0
	engine.gs.Board.T[tileIdx].Army = 10
	engine.gs.Players[0].OwnedTiles = []int{tileIdx}
	engine.gs.Players[0].Alive = false // Dead player

	mask := engine.GetLegalActionMask(0)
	
	// All actions should be false for dead player
	for i, legal := range mask {
		assert.False(t, legal, "Dead player should have no legal moves at index %d", i)
	}
}

func TestGetLegalActionMask_InvalidPlayer(t *testing.T) {
	// Test invalid player IDs
	engine := createTestEngineForActionMask(3, 3, 2)
	require.NotNil(t, engine)

	// Test negative player ID
	mask := engine.GetLegalActionMask(-1)
	assert.Len(t, mask, 36)
	for _, legal := range mask {
		assert.False(t, legal, "Invalid player should have no legal moves")
	}
	
	// Test player ID beyond range
	mask = engine.GetLegalActionMask(5)
	assert.Len(t, mask, 36)
	for _, legal := range mask {
		assert.False(t, legal, "Invalid player should have no legal moves")
	}
}

func TestGetLegalActionMask_ComplexScenario(t *testing.T) {
	// Test a more complex scenario with multiple owned tiles
	engine := createTestEngineForActionMask(5, 5, 2)
	require.NotNil(t, engine)

	// Player 0 owns a few tiles with varying army counts
	tiles := []struct {
		x, y, army int
	}{
		{1, 1, 5},  // Can move
		{2, 1, 1},  // Cannot move (insufficient army)
		{3, 3, 3},  // Can move
		{0, 0, 2},  // Can move (corner)
	}
	
	ownedTiles := make([]int, 0, len(tiles))
	for _, t := range tiles {
		idx := engine.gs.Board.Idx(t.x, t.y)
		engine.gs.Board.T[idx].Owner = 0
		engine.gs.Board.T[idx].Army = t.army
		ownedTiles = append(ownedTiles, idx)
	}
	engine.gs.Players[0].OwnedTiles = ownedTiles
	engine.gs.Players[0].Alive = true
	
	// Add a mountain to block one move
	mountainIdx := engine.gs.Board.Idx(1, 2)
	engine.gs.Board.T[mountainIdx].Type = core.TileMountain

	mask := engine.GetLegalActionMask(0)
	
	// Count legal moves
	legalMoves := 0
	for _, legal := range mask {
		if legal {
			legalMoves++
		}
	}
	
	// Verify reasonable number of legal moves
	assert.Greater(t, legalMoves, 0, "Should have some legal moves")
	assert.Less(t, legalMoves, 20, "Should not have too many legal moves")
	
	// Verify specific blocked move (1,1) -> (1,2) due to mountain
	blockedMoveIdx := (1*5 + 1)*4 + 2 // down from (1,1)
	assert.False(t, mask[blockedMoveIdx], "Should not be able to move into mountain")
}