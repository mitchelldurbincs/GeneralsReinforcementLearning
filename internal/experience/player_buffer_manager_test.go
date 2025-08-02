package experience

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPlayerBufferManager_Creation(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	
	// Test with mutex-based buffers
	mgr := NewPlayerBufferManager(100, false, logger)
	assert.NotNil(t, mgr)
	assert.Equal(t, 100, mgr.bufferCapacity)
	assert.False(t, mgr.useLockFree)
	
	// Test with lock-free buffers
	mgr2 := NewPlayerBufferManager(200, true, logger)
	assert.NotNil(t, mgr2)
	assert.Equal(t, 200, mgr2.bufferCapacity)
	assert.True(t, mgr2.useLockFree)
	
	// Test with zero capacity (should use default)
	mgr3 := NewPlayerBufferManager(0, false, logger)
	assert.Equal(t, 1000, mgr3.bufferCapacity)
}

func TestPlayerBufferManager_GetOrCreateBuffer(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	mgr := NewPlayerBufferManager(100, false, logger)
	
	// Create buffer for player 1
	buf1, err := mgr.GetOrCreateBuffer(1)
	require.NoError(t, err)
	assert.NotNil(t, buf1)
	assert.Equal(t, int32(1), atomic.LoadInt32(&mgr.activeBuffers))
	
	// Get same buffer again
	buf1Again, err := mgr.GetOrCreateBuffer(1)
	require.NoError(t, err)
	assert.Equal(t, buf1, buf1Again) // Should be same instance
	assert.Equal(t, int32(1), atomic.LoadInt32(&mgr.activeBuffers))
	
	// Create buffer for player 2
	buf2, err := mgr.GetOrCreateBuffer(2)
	require.NoError(t, err)
	assert.NotNil(t, buf2)
	assert.Equal(t, int32(2), atomic.LoadInt32(&mgr.activeBuffers))
}

func TestPlayerBufferManager_AddAndGetExperiences(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	mgr := NewPlayerBufferManager(100, false, logger)
	
	// Add experiences for different players
	exp1 := &experiencepb.Experience{ExperienceId: "p1-1", PlayerId: 1}
	exp2 := &experiencepb.Experience{ExperienceId: "p1-2", PlayerId: 1}
	exp3 := &experiencepb.Experience{ExperienceId: "p2-1", PlayerId: 2}
	
	err := mgr.AddExperience(1, exp1)
	require.NoError(t, err)
	err = mgr.AddExperience(1, exp2)
	require.NoError(t, err)
	err = mgr.AddExperience(2, exp3)
	require.NoError(t, err)
	
	assert.Equal(t, uint64(3), atomic.LoadUint64(&mgr.totalExperiences))
	
	// Get experiences for player 1
	p1Exps, err := mgr.GetExperiences(1, 10)
	require.NoError(t, err)
	assert.Len(t, p1Exps, 2)
	
	// Get experiences for player 2
	p2Exps, err := mgr.GetExperiences(2, 10)
	require.NoError(t, err)
	assert.Len(t, p2Exps, 1)
	
	// Try to get for non-existent player
	_, err = mgr.GetExperiences(3, 10)
	assert.Error(t, err)
}

func TestPlayerBufferManager_GetAllExperiences(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	mgr := NewPlayerBufferManager(100, true, logger) // Use lock-free
	
	// Add experiences for multiple players
	for p := int32(1); p <= 3; p++ {
		for i := 0; i < 5; i++ {
			exp := &experiencepb.Experience{
				ExperienceId: string(rune('A'+p-1)) + string(rune('0'+i)),
				PlayerId:     p,
			}
			err := mgr.AddExperience(p, exp)
			require.NoError(t, err)
		}
	}
	
	// Get all experiences grouped by player
	allExps := mgr.GetAllExperiences()
	assert.Len(t, allExps, 3) // 3 players
	for p := int32(1); p <= 3; p++ {
		assert.Len(t, allExps[p], 5)
	}
	
	// Merge all experiences
	merged := mgr.MergeAllExperiences()
	assert.Len(t, merged, 15) // 3 players * 5 experiences
}

func TestPlayerBufferManager_RemovePlayerBuffer(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	mgr := NewPlayerBufferManager(100, false, logger)
	
	// Create buffers for players
	_, err := mgr.GetOrCreateBuffer(1)
	require.NoError(t, err)
	_, err = mgr.GetOrCreateBuffer(2)
	require.NoError(t, err)
	assert.Equal(t, int32(2), atomic.LoadInt32(&mgr.activeBuffers))
	
	// Remove player 1's buffer
	err = mgr.RemovePlayerBuffer(1)
	require.NoError(t, err)
	assert.Equal(t, int32(1), atomic.LoadInt32(&mgr.activeBuffers))
	
	// Try to get removed buffer
	_, err = mgr.GetExperiences(1, 10)
	assert.Error(t, err)
	
	// Remove non-existent buffer (should be no-op)
	err = mgr.RemovePlayerBuffer(99)
	assert.NoError(t, err)
}

func TestPlayerBufferManager_ConcurrentAccess(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	mgr := NewPlayerBufferManager(1000, true, logger) // Use lock-free
	
	const numPlayers = 10
	const experiencesPerPlayer = 100
	const numGoroutines = 5
	
	var wg sync.WaitGroup
	
	// Multiple goroutines adding experiences for each player
	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			
			for p := int32(0); p < numPlayers; p++ {
				for i := 0; i < experiencesPerPlayer/numGoroutines; i++ {
					exp := &experiencepb.Experience{
						ExperienceId: string(rune('A'+goroutineID)) + string(rune('0'+p)) + string(rune('0'+i)),
						PlayerId:     p,
					}
					err := mgr.AddExperience(p, exp)
					if err != nil {
						t.Errorf("Failed to add experience: %v", err)
					}
				}
			}
		}(g)
	}
	
	wg.Wait()
	
	// Verify results
	stats := mgr.Stats()
	assert.Equal(t, uint64(numPlayers*experiencesPerPlayer), stats.TotalExperiences)
	assert.Equal(t, numPlayers, stats.ActiveBuffers)
	
	// Verify each player has correct number of experiences
	for p := int32(0); p < numPlayers; p++ {
		assert.Equal(t, experiencesPerPlayer, stats.PlayerStats[p].BufferSize)
	}
}

func TestPlayerBufferManager_Stats(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	mgr := NewPlayerBufferManager(100, false, logger)
	
	// Mock time for consistent testing
	oldTimeNow := timeNow
	mockTime := int64(1000)
	timeNow = func() int64 { return mockTime }
	defer func() { timeNow = oldTimeNow }()
	
	// Add experiences
	_ = mgr.AddExperience(1, &experiencepb.Experience{ExperienceId: "1"})
	_ = mgr.AddExperience(1, &experiencepb.Experience{ExperienceId: "2"})
	_ = mgr.AddExperience(2, &experiencepb.Experience{ExperienceId: "3"})
	
	stats := mgr.Stats()
	assert.Equal(t, uint64(3), stats.TotalExperiences)
	assert.Equal(t, 2, stats.ActiveBuffers)
	assert.Equal(t, 3, stats.TotalBuffered)
	
	// Check player-specific stats
	assert.Equal(t, 2, stats.PlayerStats[1].BufferSize)
	assert.Equal(t, 1, stats.PlayerStats[2].BufferSize)
	assert.Equal(t, mockTime, stats.PlayerStats[1].Created)
	assert.Equal(t, mockTime, stats.PlayerStats[1].LastUsed)
}

func TestPlayerBufferManager_Close(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	mgr := NewPlayerBufferManager(100, false, logger)
	
	// Create some buffers
	_, _ = mgr.GetOrCreateBuffer(1)
	_, _ = mgr.GetOrCreateBuffer(2)
	
	// Close manager
	err := mgr.Close()
	require.NoError(t, err)
	
	// Operations should fail after close
	_, err = mgr.GetOrCreateBuffer(3)
	assert.Equal(t, ErrBufferClosed, err)
	
	err = mgr.AddExperience(1, &experiencepb.Experience{})
	assert.Equal(t, ErrBufferClosed, err)
	
	// Close again should be idempotent
	err = mgr.Close()
	assert.NoError(t, err)
}

func TestDistributedCollector_Creation(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	
	// Test with mutex-based buffers
	collector := NewDistributedCollector(100, false, "test-game", logger)
	assert.NotNil(t, collector)
	assert.Equal(t, "test-game", collector.gameID)
	assert.False(t, collector.manager.useLockFree)
	
	// Test with lock-free buffers
	collector2 := NewDistributedCollector(200, true, "test-game-2", logger)
	assert.NotNil(t, collector2)
	assert.True(t, collector2.manager.useLockFree)
}

func TestDistributedCollector_OnStateTransition(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	collector := NewDistributedCollector(100, true, "test-game", logger)
	
	// Create test states
	board := &core.Board{
		W: 10,
		H: 10,
		T: make([]core.Tile, 100),
	}
	// Set up generals
	board.T[11] = core.Tile{Owner: 1, Army: 1, Type: core.TileGeneral} // (1,1)
	board.T[88] = core.Tile{Owner: 2, Army: 1, Type: core.TileGeneral} // (8,8)
	
	prevState := &game.GameState{
		Board: board,
		Players: []game.Player{
			{ID: 0, Alive: false}, // Placeholder for 0-indexed
			{ID: 1, Alive: true, GeneralIdx: 11},
			{ID: 2, Alive: true, GeneralIdx: 88},
		},
		Turn: 10,
	}
	
	currState := &game.GameState{
		Board: board.Clone(),
		Players: []game.Player{
			{ID: 0, Alive: false}, // Placeholder for 0-indexed
			{ID: 1, Alive: true, GeneralIdx: 11},
			{ID: 2, Alive: true, GeneralIdx: 88},
		},
		Turn: 11,
	}
	
	// Create actions
	actions := map[int]*game.Action{
		1: {Type: game.ActionTypeMove, From: core.Coordinate{X: 1, Y: 1}, To: core.Coordinate{X: 1, Y: 2}},
		2: {Type: game.ActionTypeMove, From: core.Coordinate{X: 8, Y: 8}, To: core.Coordinate{X: 8, Y: 7}},
	}
	
	// Collect experiences
	collector.OnStateTransition(prevState, currState, actions)
	
	// Wait a bit for goroutines to complete
	time.Sleep(10 * time.Millisecond)
	
	// Verify experiences were collected
	allExps := collector.GetExperiences()
	assert.Len(t, allExps, 2) // One for each player
	
	// Check player-specific experiences
	p1Exps, err := collector.GetPlayerExperiences(1)
	require.NoError(t, err)
	assert.Len(t, p1Exps, 1)
	assert.Equal(t, int32(1), p1Exps[0].PlayerId)
	
	p2Exps, err := collector.GetPlayerExperiences(2)
	require.NoError(t, err)
	assert.Len(t, p2Exps, 1)
	assert.Equal(t, int32(2), p2Exps[0].PlayerId)
}

func TestDistributedCollector_Clear(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	collector := NewDistributedCollector(100, false, "test-game", logger)
	
	// Add some experiences
	_ = collector.manager.AddExperience(1, &experiencepb.Experience{ExperienceId: "1"})
	_ = collector.manager.AddExperience(2, &experiencepb.Experience{ExperienceId: "2"})
	
	assert.Equal(t, 2, collector.GetExperienceCount())
	
	// Clear
	collector.Clear()
	
	assert.Equal(t, 0, collector.GetExperienceCount())
	assert.Equal(t, int32(0), atomic.LoadInt32(&collector.manager.activeBuffers))
}

func BenchmarkPlayerBufferManager_AddExperience(b *testing.B) {
	logger := zerolog.Nop()
	mgr := NewPlayerBufferManager(8192, true, logger) // Use lock-free
	exp := &experiencepb.Experience{ExperienceId: "bench", Reward: 1.0}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		playerID := int32(0)
		for pb.Next() {
			// Round-robin across 10 players
			playerID = (playerID + 1) % 10
			_ = mgr.AddExperience(playerID, exp)
		}
	})
	
	b.StopTimer()
	stats := mgr.Stats()
	b.ReportMetric(float64(stats.TotalExperiences)/b.Elapsed().Seconds(), "adds/sec")
}

func BenchmarkDistributedCollector_OnStateTransition(b *testing.B) {
	logger := zerolog.Nop()
	collector := NewDistributedCollector(8192, true, "bench-game", logger)
	
	// Create test states
	board := &core.Board{
		W: 20,
		H: 20,
		T: make([]core.Tile, 400),
	}
	// Set up generals for players
	for i := 1; i <= 4; i++ {
		idx := i*2*20 + i*2 // Position at (i*2, i*2)
		board.T[idx] = core.Tile{Owner: i, Army: 1, Type: core.TileGeneral}
	}
	
	players := []game.Player{{ID: 0, Alive: false}} // 0-indexed placeholder
	for i := 1; i <= 4; i++ {
		players = append(players, game.Player{
			ID:         i,
			Alive:      true,
			GeneralIdx: i*2*20 + i*2,
		})
	}
	
	prevState := &game.GameState{
		Board:   board,
		Players: players,
		Turn:    10,
	}
	
	currState := &game.GameState{
		Board:   board.Clone(),
		Players: players,
		Turn:    11,
	}
	
	// Create actions for all players
	actions := make(map[int]*game.Action)
	for i := 1; i <= 4; i++ {
		actions[i] = &game.Action{
			Type: game.ActionTypeMove,
			From: core.Coordinate{X: i * 2, Y: i * 2},
			To:   core.Coordinate{X: i*2 + 1, Y: i * 2},
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		collector.OnStateTransition(prevState, currState, actions)
	}
	
	b.StopTimer()
	totalExps := collector.manager.Stats().TotalExperiences
	b.ReportMetric(float64(totalExps)/b.Elapsed().Seconds(), "experiences/sec")
}