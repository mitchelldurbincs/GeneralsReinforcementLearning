package experience

import (
	"testing"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/stretchr/testify/assert"
)

// createTestGameStateWithDetails creates a more detailed test game state for serializer tests
func createTestGameStateWithDetails(width, height int) *game.GameState {
	state := createTestGameState(width, height)
	
	// Add player 1
	if len(state.Players) == 1 {
		lastIdx := width*height - 1
		state.Board.T[lastIdx] = core.Tile{
			Type:  core.TileGeneral,
			Owner: 1,
			Army:  8,
		}
		state.Players = append(state.Players, game.Player{
			ID:         1,
			Alive:      true,
			ArmyCount:  8,
			GeneralIdx: lastIdx,
			OwnedTiles: []int{lastIdx},
		})
	}
	
	// Add some owned territories
	state.Board.T[1] = core.Tile{Type: core.TileNormal, Owner: 0, Army: 5}
	state.Board.T[2] = core.Tile{Type: core.TileNormal, Owner: 0, Army: 3}
	
	// City at (2, 2)
	if width > 2 && height > 2 {
		cityIdx := 2*width + 2
		state.Board.T[cityIdx] = core.Tile{
			Type:  core.TileCity,
			Owner: -1,
			Army:  40,
		}
	}
	
	// Mountain at (1, 1)
	if width > 1 && height > 1 {
		mountainIdx := 1*width + 1
		state.Board.T[mountainIdx] = core.Tile{
			Type: core.TileMountain,
		}
	}
	
	// Make all tiles visible for both players by default
	for i := range state.Board.T {
		for playerID := 0; playerID < len(state.Players); playerID++ {
			state.Board.T[i].SetVisible(playerID, true)
		}
	}
	
	return state
}

func TestStateToTensor(t *testing.T) {
	serializer := NewSerializer()
	state := createTestGameStateWithDetails(5, 5)
	
	// Test serialization for player 0
	tensor := serializer.StateToTensor(state, 0)
	
	// Check tensor size
	expectedSize := NumChannels * 5 * 5
	assert.Equal(t, expectedSize, len(tensor), "Tensor size mismatch")
	
	// Check specific values
	// Player 0's general at (0,0) should have normalized armies in channel 0
	generalIdx := serializer.getChannelIndex(ChannelOwnArmies, 0, 0, 5, 5)
	assert.Equal(t, float32(10.0/MaxArmyValue), tensor[generalIdx], "General armies not properly normalized")
	
	// Player 0's territory at (0,0) should be marked in channel 2
	territoryIdx := serializer.getChannelIndex(ChannelOwnTerritory, 0, 0, 5, 5)
	assert.Equal(t, float32(1.0), tensor[territoryIdx], "Own territory not marked")
	
	// Mountain at (1,1) should be marked in channel 6
	mountainIdx := serializer.getChannelIndex(ChannelMountains, 1, 1, 5, 5)
	assert.Equal(t, float32(1.0), tensor[mountainIdx], "Mountain not marked")
	
	// City at (2,2) should be marked in channel 5
	cityIdx := serializer.getChannelIndex(ChannelCities, 2, 2, 5, 5)
	assert.Equal(t, float32(1.0), tensor[cityIdx], "City not marked")
	
	// Neutral city should be marked as neutral territory
	neutralIdx := serializer.getChannelIndex(ChannelNeutralTerritory, 2, 2, 5, 5)
	assert.Equal(t, float32(1.0), tensor[neutralIdx], "Neutral territory not marked")
}

func TestGenerateActionMask(t *testing.T) {
	serializer := NewSerializer()
	state := createTestGameStateWithDetails(3, 3)
	
	// Test action mask for player 0
	mask := serializer.GenerateActionMask(state, 0)
	
	// Expected number of actions: 3x3 tiles * 4 directions = 36
	assert.Equal(t, 36, len(mask), "Action mask size mismatch")
	
	// Player 0 can move from (0,0) with 10 armies
	// Can move right and down (not up or left - edge of board)
	rightIdx := serializer.actionToFlatIndex(0, 0, 3, 3) // right
	downIdx := serializer.actionToFlatIndex(0, 0, 1, 3)  // down
	
	assert.True(t, mask[rightIdx], "Should be able to move right from (0,0)")
	assert.True(t, mask[downIdx], "Should be able to move down from (0,0)")
	
	// Cannot move up or left from (0,0) - board edges
	upIdx := serializer.actionToFlatIndex(0, 0, 0, 3)   // up
	leftIdx := serializer.actionToFlatIndex(0, 0, 2, 3) // left
	assert.False(t, mask[upIdx], "Should not be able to move up from (0,0)")
	assert.False(t, mask[leftIdx], "Should not be able to move left from (0,0)")
	
	// Cannot move into mountain at (1,1) from (1,0)
	mountainMoveIdx := serializer.actionToFlatIndex(1, 0, 1, 3) // down into mountain
	assert.False(t, mask[mountainMoveIdx], "Should not be able to move into mountain")
}

func TestActionToIndex(t *testing.T) {
	serializer := NewSerializer()
	
	testCases := []struct {
		name      string
		action    *game.Action
		boardWidth int
		expected  int
	}{
		{
			name: "Move up from (1,1)",
			action: &game.Action{
				Type: game.ActionTypeMove,
				From: core.Coordinate{X: 1, Y: 1},
				To:   core.Coordinate{X: 1, Y: 0},
			},
			boardWidth: 5,
			expected:  (1*5+1)*4 + 0,
		},
		{
			name: "Move right from (2,3)",
			action: &game.Action{
				Type: game.ActionTypeMove,
				From: core.Coordinate{X: 2, Y: 3},
				To:   core.Coordinate{X: 3, Y: 3},
			},
			boardWidth: 5,
			expected:  (3*5+2)*4 + 3,
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			idx := serializer.ActionToIndex(tc.action, tc.boardWidth)
			assert.Equal(t, tc.expected, idx)
		})
	}
}

func TestIndexToAction(t *testing.T) {
	serializer := NewSerializer()
	
	// Test round-trip conversion
	originalFromX, originalFromY := 2, 3
	originalToX, originalToY := 1, 3 // Left move
	boardWidth, boardHeight := 5, 5
	
	// Convert to index
	idx := serializer.actionToFlatIndex(originalFromX, originalFromY, 2, boardWidth) // 2 = Left
	
	// Convert back
	fromX, fromY, toX, toY := serializer.IndexToAction(idx, boardWidth, boardHeight)
	
	assert.Equal(t, originalFromX, fromX)
	assert.Equal(t, originalFromY, fromY)
	assert.Equal(t, originalToX, toX)
	assert.Equal(t, originalToY, toY)
}

func TestFogOfWar(t *testing.T) {
	serializer := NewSerializer()
	state := createTestGameStateWithDetails(3, 3)
	
	// Enable fog of war
	state.FogOfWarEnabled = true
	
	// Reset visibility - only tiles around owned territories are visible
	for i := range state.Board.T {
		for playerID := 0; playerID < len(state.Players); playerID++ {
			state.Board.T[i].SetVisible(playerID, false)
		}
	}
	
	// Make tiles around (0,0) visible for player 0
	state.Board.T[0].SetVisible(0, true) // (0,0)
	state.Board.T[1].SetVisible(0, true) // (1,0)
	state.Board.T[3].SetVisible(0, true) // (0,1)
	if len(state.Board.T) > 4 {
		state.Board.T[4].SetVisible(0, true) // (1,1)
	}
	
	tensor := serializer.StateToTensor(state, 0)
	
	// Check visible tiles
	visibleIdx00 := serializer.getChannelIndex(ChannelVisible, 0, 0, 3, 3)
	assert.Equal(t, float32(1.0), tensor[visibleIdx00], "Tile (0,0) should be visible")
	
	// Check fog tiles
	fogIdx22 := serializer.getChannelIndex(ChannelFog, 2, 2, 3, 3)
	assert.Equal(t, float32(1.0), tensor[fogIdx22], "Tile (2,2) should be in fog")
	
	// Enemy general at (2,2) should not have army information due to fog
	enemyArmyIdx := serializer.getChannelIndex(ChannelEnemyArmies, 2, 2, 3, 3)
	assert.Equal(t, float32(0.0), tensor[enemyArmyIdx], "Enemy armies should not be visible in fog")
}

func TestNormalizeArmyValue(t *testing.T) {
	serializer := NewSerializer()
	
	testCases := []struct {
		armies   int
		expected float32
	}{
		{0, 0.0},
		{100, 0.1},
		{500, 0.5},
		{1000, 1.0},
		{2000, 1.0}, // Should cap at 1.0
	}
	
	for _, tc := range testCases {
		normalized := serializer.NormalizeArmyValue(tc.armies)
		assert.Equal(t, tc.expected, normalized, "Army normalization failed for %d armies", tc.armies)
	}
}

func BenchmarkStateToTensor(b *testing.B) {
	serializer := NewSerializer()
	state := createTestGameState(20, 20)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = serializer.StateToTensor(state, 0)
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "states/sec")
}

func BenchmarkGenerateActionMask(b *testing.B) {
	serializer := NewSerializer()
	state := createTestGameState(20, 20)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = serializer.GenerateActionMask(state, 0)
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "masks/sec")
}