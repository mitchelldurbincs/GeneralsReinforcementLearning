package game

import (
	"math/rand"
	"testing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

func TestEngine_NewEngine_CreatesValidGame(t *testing.T) {
	// Arrange
	seed := int64(12345)
	rng := rand.New(rand.NewSource(seed))
	
	// Act
	engine := NewEngine(8, 8, 2, rng)
	
	// Assert
	state := engine.GameState()
	assert.Equal(t, 0, state.Turn, "Game should start at turn 0")
	assert.Len(t, state.Players, 2, "Should have 2 players")
	assert.Equal(t, 8, state.Board.W, "Board width should be 8")
	assert.Equal(t, 8, state.Board.H, "Board height should be 8")
	assert.False(t, engine.IsGameOver(), "Game should not be over initially")
	
	// Check that both players have generals
	for i, player := range state.Players {
		assert.True(t, player.Alive, "Player %d should be alive", i)
		assert.True(t, player.GeneralIdx >= 0, "Player %d should have a general", i)
		assert.True(t, player.ArmyCount > 0, "Player %d should have armies", i)
	}
}

func TestEngine_Step_WithNoActions_AppliesProduction(t *testing.T) {
	// Arrange
	engine := NewEngine(5, 5, 2, rand.New(rand.NewSource(42)))
	initialState := engine.GameState()
	
	// Act
	err := engine.Step([]core.Action{})
	
	// Assert
	require.NoError(t, err, "Step with no actions should not error")
	
	newState := engine.GameState()
	assert.Equal(t, initialState.Turn+1, newState.Turn, "Turn should increment")
	
	// Check that generals got production
	for i, player := range newState.Players {
		initialPlayer := initialState.Players[i]
		assert.True(t, player.ArmyCount > initialPlayer.ArmyCount, 
			"Player %d should have more armies after production", i)
	}
}

func TestEngine_Step_WithValidActions_ProcessesCorrectly(t *testing.T) {
	// Arrange
	engine := NewEngine(5, 5, 2, rand.New(rand.NewSource(42)))
	
	// Find a valid move for player 0
	state := engine.GameState()
	var validAction *core.MoveAction
	
	for y := 0; y < state.Board.H; y++ {
		for x := 0; x < state.Board.W; x++ {
			tile := state.Board.T[state.Board.Idx(x, y)]
			if tile.Owner == 0 && tile.Army > 1 {
				// Try moving right
				if x+1 < state.Board.W {
					action := &core.MoveAction{
						PlayerID: 0,
						FromX: x, FromY: y,
						ToX: x+1, ToY: y,
						MoveAll: false,
					}
					if action.Validate(state.Board, 0) == nil {
						validAction = action
						break
					}
				}
			}
		}
		if validAction != nil {
			break
		}
	}
	
	require.NotNil(t, validAction, "Should find at least one valid action")
	
	// Act
	err := engine.Step([]core.Action{validAction})
	
	// Assert
	require.NoError(t, err, "Step with valid action should not error")
	
	newState := engine.GameState()
	assert.Equal(t, state.Turn+1, newState.Turn, "Turn should increment")
}

// Test complete game scenario
func TestEngine_CompleteGame_DeterministicOutcome(t *testing.T) {
	// This is a "golden test" - with the same seed, should always produce same result
	seed := int64(98765)
	
	// Run the same game twice
	result1 := runCompleteGame(t, seed)
	result2 := runCompleteGame(t, seed)
	
	// Should be identical
	assert.Equal(t, result1.winner, result2.winner, "Same seed should produce same winner")
	assert.Equal(t, result1.finalTurn, result2.finalTurn, "Same seed should produce same game length")
	assert.Equal(t, result1.finalArmyCounts, result2.finalArmyCounts, "Same seed should produce same final armies")
}

type gameResult struct {
	winner          int
	finalTurn       int
	finalArmyCounts []int
}

func runCompleteGame(t *testing.T, seed int64) gameResult {
	engine := NewEngine(6, 6, 2, rand.New(rand.NewSource(seed)))
	rng := rand.New(rand.NewSource(seed + 1)) // Different seed for actions
	
	maxTurns := 100
	for turn := 0; turn < maxTurns && !engine.IsGameOver(); turn++ {
		// Generate some random actions
		actions := generateTestActions(engine, rng, 2) // Max 2 actions per turn
		
		err := engine.Step(actions)
		require.NoError(t, err, "Game step should not error at turn %d", turn)
	}
	
	finalState := engine.GameState()
	armyCounts := make([]int, len(finalState.Players))
	for i, player := range finalState.Players {
		armyCounts[i] = player.ArmyCount
	}
	
	return gameResult{
		winner:          engine.GetWinner(),
		finalTurn:       finalState.Turn,
		finalArmyCounts: armyCounts,
	}
}

func generateTestActions(engine *Engine, rng *rand.Rand, maxActions int) []core.Action {
	var actions []core.Action
	state := engine.GameState()
	
	for _, player := range state.Players {
		if !player.Alive || len(actions) >= maxActions {
			continue
		}
		
		// 30% chance to make an action
		if rng.Float32() > 0.3 {
			continue
		}
		
		// Find a valid move
		for attempts := 0; attempts < 10; attempts++ {
			x, y := rng.Intn(state.Board.W), rng.Intn(state.Board.H)
			tile := state.Board.T[state.Board.Idx(x, y)]
			
			if tile.Owner != player.ID || tile.Army <= 1 {
				continue
			}
			
			// Try a random direction
			directions := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
			dir := directions[rng.Intn(4)]
			toX, toY := x+dir[0], y+dir[1]
			
			if toX < 0 || toX >= state.Board.W || toY < 0 || toY >= state.Board.H {
				continue
			}
			
			action := &core.MoveAction{
				PlayerID: player.ID,
				FromX: x, FromY: y,
				ToX: toX, ToY: toY,
				MoveAll: rng.Float32() < 0.7,
			}
			
			if action.Validate(state.Board, player.ID) == nil {
				actions = append(actions, action)
				break
			}
		}
	}
	
	return actions
}