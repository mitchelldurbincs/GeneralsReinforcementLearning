package experience

import (
	"testing"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
)

func TestSimpleCollector_Creation(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(100, "test-game-123", logger)
	
	assert.NotNil(t, collector)
	assert.Equal(t, 100, collector.maxSize)
	assert.Equal(t, "test-game-123", collector.gameID)
	assert.Equal(t, 0, len(collector.experiences))
}

func TestSimpleCollector_OnStateTransition(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(100, "test-game-123", logger)
	
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	// Simulate a move
	actions := map[int]*game.Action{
		0: {
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: 0, Y: 0},
			To:   core.Coordinate{X: 1, Y: 0},
		},
	}
	
	// Make state changes
	currState.Board.T[1].Owner = 0
	currState.Board.T[1].Army = 5
	currState.Turn = 2
	
	collector.OnStateTransition(prevState, currState, actions)
	
	// Check experience was collected
	assert.Equal(t, 1, collector.GetExperienceCount())
	
	experiences := collector.GetExperiences()
	assert.Len(t, experiences, 1)
	
	exp := experiences[0]
	assert.Equal(t, "test-game-123", exp.GameId)
	assert.Equal(t, int32(0), exp.PlayerId)
	assert.Equal(t, int32(2), exp.Turn)
	assert.NotNil(t, exp.State)
	assert.NotNil(t, exp.NextState)
	assert.NotNil(t, exp.ActionMask)
	assert.False(t, exp.Done)
}

func TestSimpleCollector_MultipleActions(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(100, "test-game-123", logger)
	
	prevState := createTestGameState(5, 5)
	currState := createTestGameState(5, 5)
	
	// Multiple players take actions
	actions := map[int]*game.Action{
		0: {
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: 0, Y: 0},
			To:   core.Coordinate{X: 1, Y: 0},
		},
		1: {
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: 4, Y: 4},
			To:   core.Coordinate{X: 3, Y: 4},
		},
	}
	
	collector.OnStateTransition(prevState, currState, actions)
	
	// Should have one experience per action
	assert.Equal(t, 2, collector.GetExperienceCount())
	
	experiences := collector.GetExperiences()
	
	// Check player IDs
	playerIDs := map[int32]bool{}
	for _, exp := range experiences {
		playerIDs[exp.PlayerId] = true
	}
	assert.True(t, playerIDs[0])
	assert.True(t, playerIDs[1])
}

func TestSimpleCollector_BufferOverflow(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(2, "test-game-123", logger) // Small buffer
	
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	actions := map[int]*game.Action{
		0: {
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: 0, Y: 0},
			To:   core.Coordinate{X: 1, Y: 0},
		},
	}
	
	// Fill buffer
	collector.OnStateTransition(prevState, currState, actions)
	collector.OnStateTransition(prevState, currState, actions)
	
	assert.Equal(t, 2, collector.GetExperienceCount())
	
	// Try to add one more - should be dropped
	collector.OnStateTransition(prevState, currState, actions)
	assert.Equal(t, 2, collector.GetExperienceCount()) // Still 2
}

func TestSimpleCollector_OnGameEnd(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(100, "test-game-123", logger)
	
	finalState := createTestGameState(3, 3)
	// Set up a winning state by eliminating other players
	finalState.Players = append(finalState.Players, game.Player{
		ID:         1,
		Alive:      false,
		ArmyCount:  0,
		GeneralIdx: -1,
	})
	finalState.Turn = 50
	
	// Should not panic and should log appropriately
	collector.OnGameEnd(finalState)
}

func TestSimpleCollector_GameEndingExperience(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(100, "test-game-123", logger)
	
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	// Game ends - eliminate player 1
	currState.Players[1] = game.Player{
		ID:         1,
		Alive:      false,
		ArmyCount:  0,
		GeneralIdx: -1,
	}
	
	actions := map[int]*game.Action{
		0: {
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: 0, Y: 0},
			To:   core.Coordinate{X: 1, Y: 0},
		},
	}
	
	collector.OnStateTransition(prevState, currState, actions)
	
	experiences := collector.GetExperiences()
	assert.Len(t, experiences, 1)
	assert.True(t, experiences[0].Done)
	
	// Check reward is win/loss reward
	assert.Equal(t, float32(1.0), experiences[0].Reward) // Winner gets +1.0
}

func TestSimpleCollector_Clear(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(100, "test-game-123", logger)
	
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	actions := map[int]*game.Action{
		0: {
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: 0, Y: 0},
			To:   core.Coordinate{X: 1, Y: 0},
		},
	}
	
	collector.OnStateTransition(prevState, currState, actions)
	assert.Equal(t, 1, collector.GetExperienceCount())
	
	collector.Clear()
	assert.Equal(t, 0, collector.GetExperienceCount())
}

func TestSimpleCollector_GetLatestExperiences(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(100, "test-game-123", logger)
	
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	actions := map[int]*game.Action{
		0: {
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: 0, Y: 0},
			To:   core.Coordinate{X: 1, Y: 0},
		},
	}
	
	// Add 5 experiences with different turns
	for i := 0; i < 5; i++ {
		currState.Turn = i + 1
		collector.OnStateTransition(prevState, currState, actions)
	}
	
	// Get latest 3
	latest := collector.GetLatestExperiences(3)
	assert.Len(t, latest, 3)
	
	// Should be turns 3, 4, 5 (latest)
	assert.Equal(t, int32(3), latest[0].Turn)
	assert.Equal(t, int32(4), latest[1].Turn)
	assert.Equal(t, int32(5), latest[2].Turn)
	
	// Request more than available
	all := collector.GetLatestExperiences(10)
	assert.Len(t, all, 5)
}

func TestSimpleCollector_ActionConversion(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(100, "test-game-123", logger)
	
	prevState := createTestGameState(5, 5)
	currState := createTestGameState(5, 5)
	
	// Test different directions
	testCases := []struct {
		fromX, fromY int
		toX, toY     int
	}{
		{2, 2, 2, 1}, // Up
		{2, 2, 2, 3}, // Down
		{2, 2, 1, 2}, // Left
		{2, 2, 3, 2}, // Right
	}
	
	for _, tc := range testCases {
		actions := map[int]*game.Action{
			0: {
				Type: game.ActionTypeMove,
				From: core.Coordinate{X: tc.fromX, Y: tc.fromY},
				To:   core.Coordinate{X: tc.toX, Y: tc.toY},
			},
		}
		
		// Ensure tile is owned and has armies
		tileIdx := tc.fromY*5 + tc.fromX
		prevState.Board.T[tileIdx].Owner = 0
		prevState.Board.T[tileIdx].Army = 10
		
		collector.OnStateTransition(prevState, currState, actions)
	}
	
	experiences := collector.GetExperiences()
	assert.Len(t, experiences, 4)
	
	// Each should have a different action index
	actionIndices := map[int32]bool{}
	for _, exp := range experiences {
		actionIndices[exp.Action] = true
	}
	assert.Len(t, actionIndices, 4)
}

func TestSimpleCollector_ThreadSafety(t *testing.T) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(1000, "test-game-123", logger)
	
	prevState := createTestGameState(3, 3)
	currState := createTestGameState(3, 3)
	
	actions := map[int]*game.Action{
		0: {
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: 0, Y: 0},
			To:   core.Coordinate{X: 1, Y: 0},
		},
	}
	
	// Run concurrent operations
	done := make(chan bool, 3)
	
	// Writer 1
	go func() {
		for i := 0; i < 100; i++ {
			collector.OnStateTransition(prevState, currState, actions)
		}
		done <- true
	}()
	
	// Writer 2
	go func() {
		for i := 0; i < 100; i++ {
			collector.OnStateTransition(prevState, currState, actions)
		}
		done <- true
	}()
	
	// Reader
	go func() {
		for i := 0; i < 100; i++ {
			_ = collector.GetExperiences()
			_ = collector.GetExperienceCount()
		}
		done <- true
	}()
	
	// Wait for all goroutines
	for i := 0; i < 3; i++ {
		<-done
	}
	
	// Should have collected some experiences without crashing
	count := collector.GetExperienceCount()
	assert.Greater(t, count, 0)
	assert.LessOrEqual(t, count, 200)
}

// Helper function to create a test game state
func createTestGameState(width, height int) *game.GameState {
	// Create board
	board := &core.Board{
		W: width,
		H: height,
		T: make([]core.Tile, width*height),
	}
	
	// Initialize tiles
	for i := range board.T {
		board.T[i] = core.Tile{
			Type:  core.TileNormal,
			Owner: -1,
			Army:  0,
		}
	}
	
	// Place a general for player 0 at (0, 0)
	board.T[0] = core.Tile{
		Type:  core.TileGeneral,
		Owner: 0,
		Army:  10,
	}
	
	// Place a general for player 1 at the opposite corner
	lastIdx := width*height - 1
	board.T[lastIdx] = core.Tile{
		Type:  core.TileGeneral,
		Owner: 1,
		Army:  10,
	}
	
	// Create player states
	players := []game.Player{
		{
			ID:         0,
			Alive:      true,
			ArmyCount:  10,
			GeneralIdx: 0,
			OwnedTiles: []int{0},
		},
		{
			ID:         1,
			Alive:      true,
			ArmyCount:  10,
			GeneralIdx: lastIdx,
			OwnedTiles: []int{lastIdx},
		},
	}
	
	// Create game state
	return &game.GameState{
		Board:           board,
		Players:         players,
		Turn:            1,
		FogOfWarEnabled: false,
		ChangedTiles:    make(map[int]struct{}),
		VisibilityChangedTiles: make(map[int]struct{}),
	}
}