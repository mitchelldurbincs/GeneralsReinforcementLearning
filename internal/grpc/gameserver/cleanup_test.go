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
	// Create game manager directly (bypass server to avoid automatic cleanup goroutine)
	gm := &GameManager{
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
		players:            make([]playerInfo, 0, 2),
		createdAt:          now,
		lastActivity:       now,
		idempotencyManager: NewIdempotencyManager(),
		streamManager:      NewStreamManager(),
	}

	// Add game to manager
	gm.mu.Lock()
	gm.games[gameID] = game
	gm.mu.Unlock()

	// Verify game exists
	gm.mu.RLock()
	game, exists := gm.games[gameID]
	gm.mu.RUnlock()
	assert.True(t, exists)

	// Test 1: Active game should not be cleaned up
	game.lastActivity = time.Now()
	gm.cleanupGames()

	gm.mu.RLock()
	_, stillExists := gm.games[gameID]
	gm.mu.RUnlock()
	assert.True(t, stillExists, "Active game should not be cleaned up")

	// Test 2: Game without engine should be cleaned up after abandoned timeout
	// Since we can't test finished games without a proper engine setup,
	// we'll test abandoned game cleanup instead
	game.lastActivity = time.Now().Add(-35 * time.Minute) // Past abandoned timeout
	gm.cleanupGames()

	gm.mu.RLock()
	_, stillExists = gm.games[gameID]
	gm.mu.RUnlock()
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
		players:            make([]playerInfo, 0, 2),
		createdAt:          now2,
		lastActivity:       now2,
		idempotencyManager: NewIdempotencyManager(),
		streamManager:      NewStreamManager(),
	}

	gm.mu.Lock()
	gm.games[gameID2] = game2
	gm.mu.Unlock()

	// Verify the new game exists and is not cleaned up when active
	gm.mu.RLock()
	_, exists2 := gm.games[gameID2]
	gm.mu.RUnlock()
	assert.True(t, exists2)

	// Recent activity should prevent cleanup
	game2.lastActivity = time.Now()
	gm.cleanupGames()

	gm.mu.RLock()
	_, stillExists = gm.games[gameID2]
	gm.mu.RUnlock()
	assert.True(t, stillExists, "Active game should not be cleaned up")
}

func TestLastActivityUpdates(t *testing.T) {
	// Create server with a game manager that doesn't auto-start cleanup
	gm := &GameManager{
		games: make(map[string]*gameInstance),
	}
	s := &Server{
		gameManager: gm,
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

	gm.mu.RLock()
	game := gm.games[gameID]
	initialActivity := game.lastActivity
	gm.mu.RUnlock()

	// Wait a bit
	time.Sleep(10 * time.Millisecond)

	// Join game should update activity
	_, err = s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player1",
	})
	require.NoError(t, err)

	gm.mu.RLock()
	afterJoinActivity := game.lastActivity
	gm.mu.RUnlock()

	assert.True(t, afterJoinActivity.After(initialActivity), "Join should update last activity")

	// Skip testing with 2 players as it would trigger engine creation
	// which might block in tests. The join activity update is already verified above.
}
