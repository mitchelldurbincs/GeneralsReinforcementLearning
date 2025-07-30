package experience

import (
	"testing"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/stretchr/testify/assert"
)

func TestCalculateReward_WinLoss(t *testing.T) {
	// Test win condition
	// Since we can't set GameOver directly, we'll simulate a winning state
	// by having only one player alive
	t.Skip("Requires Engine integration for proper game over state")
}

func TestCalculateReward_TerritoryChanges(t *testing.T) {
	config := DefaultRewardConfig()
	
	// Set up states where player gains territory
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	// Player 0 captures tile (2, 0)
	prevState.Board.T[2].Owner = -1
	currState.Board.T[2].Owner = 0
	
	reward := CalculateRewardWithConfig(prevState, currState, 0, config)
	expectedReward := config.TerritoryGained
	assert.Equal(t, expectedReward, reward, "Should get territory gain reward")
	
	// Test territory loss
	prevState.Board.T[1].Owner = 0
	currState.Board.T[1].Owner = 1
	
	reward = CalculateRewardWithConfig(prevState, currState, 0, config)
	expectedReward = config.TerritoryGained + config.TerritoryLost // Gained 1, lost 1
	assert.Equal(t, expectedReward, reward, "Should get net territory change reward")
}

func TestCalculateReward_CityCapture(t *testing.T) {
	config := DefaultRewardConfig()
	
	prevState := createTestGameState(5, 5)
	currState := createTestGameState(5, 5)
	
	// Add a city and have player 0 capture it
	cityIdx := 2*5 + 2
	prevState.Board.T[cityIdx] = core.Tile{
		Type: core.TileCity, Owner: -1, Army: 40,
	}
	currState.Board.T[cityIdx] = core.Tile{
		Type: core.TileCity, Owner: 0, Army: 40,
	}
	
	reward := CalculateRewardWithConfig(prevState, currState, 0, config)
	
	// Should include city capture + territory gain
	expectedReward := config.CaptureCity + config.TerritoryGained
	assert.InDelta(t, expectedReward, reward, 0.01, "Should get city capture reward")
}

func TestCalculateReward_GeneralCapture(t *testing.T) {
	config := DefaultRewardConfig()
	
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	// Player 0 captures player 1's general
	generalIdx := 3*3 - 1
	prevState.Board.T[generalIdx] = core.Tile{
		Type: core.TileGeneral, Owner: 1, Army: 8,
	}
	currState.Board.T[generalIdx] = core.Tile{
		Type: core.TileGeneral, Owner: 0, Army: 8,
	}
	
	reward := CalculateRewardWithConfig(prevState, currState, 0, config)
	
	// Should include general capture + territory gain
	expectedReward := config.CaptureGeneral + config.TerritoryGained
	assert.InDelta(t, expectedReward, reward, 0.01, "Should get general capture reward")
}

func TestCalculateReward_ArmyChanges(t *testing.T) {
	config := DefaultRewardConfig()
	
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	// Increase armies on owned tile
	prevState.Board.T[0].Army = 10
	currState.Board.T[0].Army = 15
	
	reward := CalculateRewardWithConfig(prevState, currState, 0, config)
	
	expectedReward := 5 * config.ArmyGained // Gained 5 armies
	assert.InDelta(t, expectedReward, reward, 0.01, "Should get army gain reward")
}

func TestCalculateArmyAdvantage(t *testing.T) {
	state := createTestGameState(3, 3)
	
	// Set up army counts
	// Player 0: 10 + 5 + 3 = 18 armies
	// Player 1: 8 armies
	// Total: 26 armies
	
	advantage := calculateArmyAdvantage(state, 0)
	expected := float32(18-8) / float32(26) // (18-8)/26 ≈ 0.385
	assert.InDelta(t, expected, advantage, 0.01, "Army advantage calculation incorrect")
	
	// Test from player 1's perspective
	advantage = calculateArmyAdvantage(state, 1)
	expected = float32(8-18) / float32(26) // (8-18)/26 ≈ -0.385
	assert.InDelta(t, expected, advantage, 0.01, "Army disadvantage calculation incorrect")
}

func TestCountCityChanges(t *testing.T) {
	prevState := createTestGameState(5, 5)
	currState := createTestGameState(5, 5)
	
	// Add cities
	city1Idx := 1*5 + 1
	city2Idx := 3*5 + 3
	
	// City 1: neutral -> player 0 (gained)
	prevState.Board.T[city1Idx] = core.Tile{
		Type: core.TileCity, Owner: -1, Army: 40,
	}
	currState.Board.T[city1Idx] = core.Tile{
		Type: core.TileCity, Owner: 0, Army: 40,
	}
	
	// City 2: player 0 -> player 1 (lost)
	prevState.Board.T[city2Idx] = core.Tile{
		Type: core.TileCity, Owner: 0, Army: 40,
	}
	currState.Board.T[city2Idx] = core.Tile{
		Type: core.TileCity, Owner: 1, Army: 40,
	}
	
	gained, lost := countCityChanges(prevState, currState, 0)
	assert.Equal(t, 1, gained, "Should have gained 1 city")
	assert.Equal(t, 1, lost, "Should have lost 1 city")
}

func TestCalculatePotentialReward(t *testing.T) {
	state := createTestGameState(5, 5)
	
	// Add a city at (3, 3)
	cityIdx := 3*5 + 3
	state.Board.T[cityIdx] = core.Tile{
		Type: core.TileCity, Owner: 1, Army: 40,
	}
	
	// Calculate potential reward for capturing enemy city
	reward := CalculatePotentialReward(state, 0, 3, 3)
	config := DefaultRewardConfig()
	
	expectedReward := config.CaptureCity + config.TerritoryGained
	assert.Equal(t, expectedReward, reward, "Potential reward for city capture incorrect")
}

func TestNormalizeReward(t *testing.T) {
	testCases := []struct {
		input    float32
		expected float32
	}{
		{0.5, 0.5},
		{-0.5, -0.5},
		{2.0, 1.0},
		{-2.0, -1.0},
		{0.0, 0.0},
	}
	
	for _, tc := range testCases {
		normalized := NormalizeReward(tc.input)
		assert.Equal(t, tc.expected, normalized, "Normalization failed for input %f", tc.input)
	}
}

func TestRewardConfig(t *testing.T) {
	// Test default config
	config := DefaultRewardConfig()
	assert.Equal(t, float32(1.0), config.WinGame)
	assert.Equal(t, float32(-1.0), config.LoseGame)
	assert.Equal(t, float32(0.1), config.CaptureCity)
	
	// Test custom config - skip as it requires Engine integration
	t.Skip("Requires Engine integration for proper game over state")
}

func BenchmarkCalculateReward(b *testing.B) {
	prevState := createTestGameState(20, 20)
	currState := createTestGameState(20, 20)
	
	// Make some changes
	for i := 0; i < 10; i++ {
		currState.Board.T[i].Owner = 0
		currState.Board.T[i].Army = i + 5
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CalculateReward(prevState, currState, 0)
	}
	
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "rewards/sec")
}