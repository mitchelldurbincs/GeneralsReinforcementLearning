package experience

import (
	"testing"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEnhancedCollector_Creation(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 100
	config.PersistenceConfig.Type = PersistenceTypeNone
	
	collector, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector.Close()
	
	assert.NotNil(t, collector)
	assert.Equal(t, "test-game", collector.gameID)
	assert.Equal(t, 100, collector.buffer.Capacity())
}

func TestEnhancedCollector_BasicCollection(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 100
	config.PersistenceConfig.Type = PersistenceTypeNone
	
	collector, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector.Close()
	
	// Create test states
	board := core.NewBoard(10, 10)
	board.T[0] = core.Tile{Type: core.TileGeneral, Owner: 1, Army: 1}
	board.T[11] = core.Tile{Type: core.TileNormal, Owner: 1, Army: 5}
	
	prevState := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 1, Alive: true},
		},
		Turn: 10,
		FogOfWarEnabled: false,
	}
	
	currState := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 1, Alive: true},
		},
		Turn: 11,
		FogOfWarEnabled: false,
	}
	
	actions := map[int]*game.Action{
		1: {Type: game.ActionTypeMove, From: core.FromIndex(11, 10), To: core.FromIndex(12, 10)},
	}
	
	// Collect experience
	collector.OnStateTransition(prevState, currState, actions)
	
	// Wait for async processing
	time.Sleep(100 * time.Millisecond)
	
	// Check metrics
	metrics := collector.GetMetrics()
	assert.Equal(t, int64(1), metrics.ExperiencesCollected)
	
	// Check buffer
	assert.Equal(t, 1, collector.GetExperienceCount())
}

func TestEnhancedCollector_OverflowDropOldest(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 3
	config.PersistenceConfig.Type = PersistenceTypeNone
	config.PersistenceConfig.OverflowStrategy = OverflowStrategyDropOldest
	
	collector, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector.Close()
	
	// Create test states
	board := core.NewBoard(10, 10)
	board.T[0] = core.Tile{Type: core.TileGeneral, Owner: 1, Army: 1}
	
	state := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 1, Alive: true},
		},
		Turn: 1,
		FogOfWarEnabled: false,
	}
	
	// Fill buffer beyond capacity
	for i := 0; i < 5; i++ {
		actions := map[int]*game.Action{
			1: {Type: game.ActionTypeMove, From: core.FromIndex(0, 10), To: core.FromIndex(1, 10)},
		}
		state.Turn = i + 1
		collector.OnStateTransition(state, state, actions)
	}
	
	// Wait for processing
	time.Sleep(100 * time.Millisecond)
	
	// Buffer should be at capacity
	assert.Equal(t, 3, collector.GetExperienceCount())
	
	// Check that oldest were dropped
	bufferStats := collector.GetBufferStats()
	assert.Greater(t, bufferStats.TotalDropped, int64(0))
}

func TestEnhancedCollector_OverflowDropNewest(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 3
	config.PersistenceConfig.Type = PersistenceTypeNone
	config.PersistenceConfig.OverflowStrategy = OverflowStrategyDropNewest
	
	collector, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector.Close()
	
	// Create test states
	board := core.NewBoard(10, 10)
	board.T[0] = core.Tile{Type: core.TileGeneral, Owner: 1, Army: 1}
	
	state := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 1, Alive: true},
		},
		Turn: 1,
		FogOfWarEnabled: false,
	}
	
	// Fill buffer to capacity
	for i := 0; i < 3; i++ {
		actions := map[int]*game.Action{
			1: {Type: game.ActionTypeMove, From: core.FromIndex(0, 10), To: core.FromIndex(1, 10)},
		}
		state.Turn = i + 1
		collector.OnStateTransition(state, state, actions)
	}
	
	// Try to add more (should be dropped)
	for i := 3; i < 5; i++ {
		actions := map[int]*game.Action{
			1: {Type: game.ActionTypeMove, From: core.FromIndex(0, 10), To: core.FromIndex(1, 10)},
		}
		state.Turn = i + 1
		collector.OnStateTransition(state, state, actions)
	}
	
	// Wait for processing
	time.Sleep(100 * time.Millisecond)
	
	// Buffer should still be at capacity
	assert.Equal(t, 3, collector.GetExperienceCount())
	
	// Check metrics
	metrics := collector.GetMetrics()
	assert.Equal(t, int64(3), metrics.ExperiencesCollected)
	assert.Equal(t, int64(2), metrics.ExperiencesDropped)
}

func TestEnhancedCollector_OverflowPersist(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 3
	config.BatchSize = 2
	config.PersistenceConfig.Type = PersistenceTypeFile
	config.PersistenceConfig.BaseDir = tempDir
	config.PersistenceConfig.OverflowStrategy = OverflowStrategyPersist
	
	collector, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector.Close()
	
	// Create test states
	board := core.NewBoard(10, 10)
	board.T[0] = core.Tile{Type: core.TileGeneral, Owner: 1, Army: 1}
	
	state := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 1, Alive: true},
		},
		Turn: 1,
		FogOfWarEnabled: false,
	}
	
	// Fill buffer beyond capacity
	for i := 0; i < 5; i++ {
		actions := map[int]*game.Action{
			1: {Type: game.ActionTypeMove, From: core.FromIndex(0, 10), To: core.FromIndex(1, 10)},
		}
		state.Turn = i + 1
		collector.OnStateTransition(state, state, actions)
	}
	
	// Wait for persistence
	time.Sleep(200 * time.Millisecond)
	
	// Check metrics
	metrics := collector.GetMetrics()
	assert.Greater(t, metrics.ExperiencesPersisted, int64(0))
	assert.Greater(t, metrics.BatchesFlushed, int64(0))
	
	// Check persistence stats
	persistStats := collector.GetPersistenceStats()
	assert.Greater(t, persistStats.TotalWritten, int64(0))
}

func TestEnhancedCollector_ManualFlush(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 100
	config.PersistenceConfig.Type = PersistenceTypeFile
	config.PersistenceConfig.BaseDir = tempDir
	
	collector, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector.Close()
	
	// Create test states
	board := core.NewBoard(10, 10)
	board.T[0] = core.Tile{Type: core.TileGeneral, Owner: 1, Army: 1}
	
	state := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 1, Alive: true},
		},
		Turn: 1,
		FogOfWarEnabled: false,
	}
	
	// Add some experiences
	for i := 0; i < 5; i++ {
		actions := map[int]*game.Action{
			1: {Type: game.ActionTypeMove, From: core.FromIndex(0, 10), To: core.FromIndex(1, 10)},
		}
		state.Turn = i + 1
		collector.OnStateTransition(state, state, actions)
	}
	
	// Wait for collection
	time.Sleep(100 * time.Millisecond)
	
	// Manual flush
	err = collector.FlushToPersistence()
	assert.NoError(t, err)
	
	// Buffer should be empty
	assert.Equal(t, 0, collector.GetExperienceCount())
	
	// Check metrics
	metrics := collector.GetMetrics()
	assert.Equal(t, int64(5), metrics.ExperiencesPersisted)
}

func TestEnhancedCollector_LoadFromPersistence(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 100
	config.PersistenceConfig.Type = PersistenceTypeFile
	config.PersistenceConfig.BaseDir = tempDir
	
	// First collector - save experiences
	collector1, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	
	board := core.NewBoard(10, 10)
	board.T[0] = core.Tile{Type: core.TileGeneral, Owner: 1, Army: 1}
	
	state := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 1, Alive: true},
		},
		Turn: 1,
		FogOfWarEnabled: false,
	}
	
	// Add and persist experiences
	for i := 0; i < 3; i++ {
		actions := map[int]*game.Action{
			1: {Type: game.ActionTypeMove, From: core.FromIndex(0, 10), To: core.FromIndex(1, 10)},
		}
		state.Turn = i + 1
		collector1.OnStateTransition(state, state, actions)
	}
	
	// Close collector1 - this will flush remaining experiences
	collector1.Close()
	
	// Second collector - load experiences
	collector2, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector2.Close()
	
	// Load from persistence
	err = collector2.LoadFromPersistence(10)
	assert.NoError(t, err)
	
	// Should have loaded experiences
	assert.Equal(t, 3, collector2.GetExperienceCount())
}

func TestEnhancedCollector_GameEnd(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 100
	config.FlushInterval = 1 * time.Hour // Long interval to test game end flush
	config.PersistenceConfig.Type = PersistenceTypeFile
	config.PersistenceConfig.BaseDir = tempDir
	
	collector, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector.Close()
	
	// Create test states
	board := core.NewBoard(10, 10)
	board.T[0] = core.Tile{Type: core.TileGeneral, Owner: 1, Army: 1}
	
	state := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 1, Alive: true},
		},
		Turn: 1,
		FogOfWarEnabled: false,
	}
	
	// Add experiences
	for i := 0; i < 5; i++ {
		actions := map[int]*game.Action{
			1: {Type: game.ActionTypeMove, From: core.FromIndex(0, 10), To: core.FromIndex(1, 10)},
		}
		state.Turn = i + 1
		collector.OnStateTransition(state, state, actions)
	}
	
	// Wait for collection
	time.Sleep(100 * time.Millisecond)
	
	// Trigger game end
	collector.OnGameEnd(state)
	
	// Wait for flush
	time.Sleep(200 * time.Millisecond)
	
	// All experiences should be flushed
	assert.Equal(t, 0, collector.GetExperienceCount())
}

func TestEnhancedCollector_Metrics(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	
	config := DefaultCollectorConfig()
	config.BufferCapacity = 100
	config.MetricsEnabled = true
	config.MetricsInterval = 100 * time.Millisecond
	config.PersistenceConfig.Type = PersistenceTypeNone
	
	collector, err := NewEnhancedCollector(config, "test-game", logger)
	require.NoError(t, err)
	defer collector.Close()
	
	// Wait for at least one metrics report
	time.Sleep(150 * time.Millisecond)
	
	// Metrics should be available
	metrics := collector.GetMetrics()
	assert.NotNil(t, metrics)
}