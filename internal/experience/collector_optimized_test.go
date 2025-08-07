package experience

import (
	"testing"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
)

func TestOptimizedCollector_Creation(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	collector := NewOptimizedCollector(100, "test-game", logger)

	assert.NotNil(t, collector)
	assert.Equal(t, "test-game", collector.gameID)
	assert.Equal(t, 0, collector.GetExperienceCount())
}

func TestOptimizedCollector_OnStateTransition(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	collector := NewOptimizedCollector(100, "test-game", logger)

	// Create test states
	board := &core.Board{
		W: 10,
		H: 10,
		T: make([]core.Tile, 100),
	}

	// Set up some tiles
	board.T[11] = core.Tile{Owner: 0, Army: 5, Type: core.TileNormal}
	board.T[12] = core.Tile{Owner: 1, Army: 3, Type: core.TileNormal}

	prevState := &game.GameState{
		Board:   board,
		Players: []game.Player{{ID: 0, Alive: true}, {ID: 1, Alive: true}},
		Turn:    10,
	}

	currState := &game.GameState{
		Board:   board.Clone(),
		Players: []game.Player{{ID: 0, Alive: true}, {ID: 1, Alive: true}},
		Turn:    11,
	}

	// Create actions
	actions := map[int]*game.Action{
		0: {Type: game.ActionTypeMove, From: core.Coordinate{X: 1, Y: 1}, To: core.Coordinate{X: 1, Y: 2}},
	}

	// Collect experiences
	collector.OnStateTransition(prevState, currState, actions)

	// Verify experience was collected
	assert.Equal(t, 1, collector.GetExperienceCount())

	experiences := collector.GetExperiences()
	assert.Len(t, experiences, 1)

	exp := experiences[0]
	assert.Equal(t, "test-game", exp.GameId)
	assert.Equal(t, int32(0), exp.PlayerId)
	assert.Equal(t, int32(11), exp.Turn)
	assert.Equal(t, "optimized-1.0.0", exp.Metadata["collector_version"])
}

func TestOptimizedCollector_Performance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	logger := zerolog.Nop()
	collector := NewOptimizedCollector(10000, "perf-test", logger)

	// Create larger test state
	board := &core.Board{
		W: 50,
		H: 50,
		T: make([]core.Tile, 2500),
	}

	// Set up some tiles with armies
	for i := 0; i < 100; i++ {
		board.T[i] = core.Tile{Owner: i % 4, Army: 10 + i%20, Type: core.TileNormal}
	}

	state := &game.GameState{
		Board:   board,
		Players: []game.Player{{ID: 0, Alive: true}, {ID: 1, Alive: true}, {ID: 2, Alive: true}, {ID: 3, Alive: true}},
		Turn:    1,
	}

	// Simulate many turns
	for turn := 0; turn < 1000; turn++ {
		prevState := state
		state = &game.GameState{
			Board:   state.Board.Clone(),
			Players: state.Players,
			Turn:    turn + 2,
		}

		// Create actions for all players
		actions := make(map[int]*game.Action)
		for p := 0; p < 4; p++ {
			if board.T[p].Army > 1 {
				actions[p] = &game.Action{
					Type: game.ActionTypeMove,
					From: core.Coordinate{X: p, Y: 0},
					To:   core.Coordinate{X: p, Y: 1},
				}
			}
		}

		collector.OnStateTransition(prevState, state, actions)
	}

	// Should have collected many experiences without running out of memory
	assert.Greater(t, collector.GetExperienceCount(), 0)

	// Test that visibility cache is cleared on game end
	collector.OnGameEnd(state)
}

func BenchmarkOptimizedCollector_OnStateTransition(b *testing.B) {
	logger := zerolog.Nop()
	collector := NewOptimizedCollector(10000, "bench-game", logger)

	// Create test states
	board := &core.Board{
		W: 20,
		H: 20,
		T: make([]core.Tile, 400),
	}

	for i := 0; i < 50; i++ {
		board.T[i] = core.Tile{Owner: i % 4, Army: 10, Type: core.TileNormal}
	}

	prevState := &game.GameState{
		Board:   board,
		Players: []game.Player{{ID: 0, Alive: true}, {ID: 1, Alive: true}},
		Turn:    10,
	}

	currState := &game.GameState{
		Board:   board.Clone(),
		Players: prevState.Players,
		Turn:    11,
	}

	actions := map[int]*game.Action{
		0: {Type: game.ActionTypeMove, From: core.Coordinate{X: 0, Y: 0}, To: core.Coordinate{X: 0, Y: 1}},
		1: {Type: game.ActionTypeMove, From: core.Coordinate{X: 1, Y: 0}, To: core.Coordinate{X: 1, Y: 1}},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		collector.OnStateTransition(prevState, currState, actions)
	}

	b.StopTimer()
	b.ReportMetric(float64(b.N*2)/b.Elapsed().Seconds(), "experiences/sec")
}

// Compare performance with original collector
func BenchmarkSimpleCollector_OnStateTransition(b *testing.B) {
	logger := zerolog.Nop()
	collector := NewSimpleCollector(10000, "bench-game", logger)

	// Create test states (same as above)
	board := &core.Board{
		W: 20,
		H: 20,
		T: make([]core.Tile, 400),
	}

	for i := 0; i < 50; i++ {
		board.T[i] = core.Tile{Owner: i % 4, Army: 10, Type: core.TileNormal}
	}

	prevState := &game.GameState{
		Board:   board,
		Players: []game.Player{{ID: 0, Alive: true}, {ID: 1, Alive: true}},
		Turn:    10,
	}

	currState := &game.GameState{
		Board:   board.Clone(),
		Players: prevState.Players,
		Turn:    11,
	}

	actions := map[int]*game.Action{
		0: {Type: game.ActionTypeMove, From: core.Coordinate{X: 0, Y: 0}, To: core.Coordinate{X: 0, Y: 1}},
		1: {Type: game.ActionTypeMove, From: core.Coordinate{X: 1, Y: 0}, To: core.Coordinate{X: 1, Y: 1}},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		collector.OnStateTransition(prevState, currState, actions)
	}

	b.StopTimer()
	b.ReportMetric(float64(b.N*2)/b.Elapsed().Seconds(), "experiences/sec")
}
