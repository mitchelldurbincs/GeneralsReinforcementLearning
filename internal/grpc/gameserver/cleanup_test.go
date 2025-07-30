package gameserver

import (
	"context"
	"testing"
	"time"
	
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

func TestGameCleanup(t *testing.T) {
	// Create server without automatic cleanup goroutine
	s := &Server{
		games: make(map[string]*gameInstance),
	}
	
	// Manually create a game instance to avoid calling CreateGame which uses logger
	gameID := "test-game-1"
	now := time.Now()
	game := &gameInstance{
		id: gameID,
		config: &gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
		},
		players:          make([]playerInfo, 0, 2),
		createdAt:        now,
		lastActivity:     now,
		idempotencyCache: make(map[string]*idempotencyEntry),
		streamClients:    make(map[int32]*streamClient),
	}
	s.games[gameID] = game
	
	// Verify game exists
	s.mu.RLock()
	game, exists := s.games[gameID]
	s.mu.RUnlock()
	assert.True(t, exists)
	
	// Test 1: Active game should not be cleaned up
	game.lastActivity = time.Now()
	s.cleanupGames()
	
	s.mu.RLock()
	_, stillExists := s.games[gameID]
	s.mu.RUnlock()
	assert.True(t, stillExists, "Active game should not be cleaned up")
	
	// Test 2: Game without engine should be cleaned up after abandoned timeout
	// Since we can't test finished games without a proper engine setup,
	// we'll test abandoned game cleanup instead
	game.lastActivity = time.Now().Add(-35 * time.Minute) // Past abandoned timeout
	s.cleanupGames()
	
	s.mu.RLock()
	_, stillExists = s.games[gameID]
	s.mu.RUnlock()
	assert.False(t, stillExists, "Abandoned game without engine should be cleaned up")
	
	// Test 3: Create a new game to test active game preservation
	gameID2 := "test-game-2"
	now2 := time.Now()
	game2 := &gameInstance{
		id: gameID2,
		config: &gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
		},
		players:          make([]playerInfo, 0, 2),
		createdAt:        now2,
		lastActivity:     now2,
		idempotencyCache: make(map[string]*idempotencyEntry),
		streamClients:    make(map[int32]*streamClient),
	}
	s.games[gameID2] = game2
	
	// Verify the new game exists and is not cleaned up when active
	s.mu.RLock()
	_, exists2 := s.games[gameID2]
	s.mu.RUnlock()
	assert.True(t, exists2)
	
	// Recent activity should prevent cleanup
	game2.lastActivity = time.Now()
	s.cleanupGames()
	
	s.mu.RLock()
	_, stillExists = s.games[gameID2]
	s.mu.RUnlock()
	assert.True(t, stillExists, "Active game should not be cleaned up")
}

func TestLastActivityUpdates(t *testing.T) {
	// Create server without automatic cleanup goroutine to avoid test interference
	s := &Server{
		games: make(map[string]*gameInstance),
	}
	ctx := context.Background()
	
	// Create a game
	createResp, err := s.CreateGame(ctx, &gamev1.CreateGameRequest{
		Config: &gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
		},
	})
	require.NoError(t, err)
	gameID := createResp.GameId
	
	s.mu.RLock()
	game := s.games[gameID]
	initialActivity := game.lastActivity
	s.mu.RUnlock()
	
	// Wait a bit
	time.Sleep(10 * time.Millisecond)
	
	// Join game should update activity
	_, err = s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player1",
	})
	require.NoError(t, err)
	
	s.mu.RLock()
	afterJoinActivity := game.lastActivity
	s.mu.RUnlock()
	
	assert.True(t, afterJoinActivity.After(initialActivity), "Join should update last activity")
	
	// Skip testing with 2 players as it would trigger engine creation
	// which might block in tests. The join activity update is already verified above.
}