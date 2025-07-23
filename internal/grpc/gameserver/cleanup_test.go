package gameserver

import (
	"context"
	"testing"
	"time"
	
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	
	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

func TestGameCleanup(t *testing.T) {
	// Create server without automatic cleanup
	s := &Server{
		games: make(map[string]*gameInstance),
	}
	
	ctx := context.Background()
	
	// Create a test game
	resp, err := s.CreateGame(ctx, &gamev1.CreateGameRequest{
		Config: &gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
		},
	})
	require.NoError(t, err)
	gameID := resp.GameId
	
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
	
	// Test 2: Finished game within TTL should not be cleaned up
	game.status = commonv1.GameStatus_GAME_STATUS_FINISHED
	game.lastActivity = time.Now().Add(-5 * time.Minute)
	s.cleanupGames()
	
	s.mu.RLock()
	_, stillExists = s.games[gameID]
	s.mu.RUnlock()
	assert.True(t, stillExists, "Recently finished game should not be cleaned up")
	
	// Test 3: Finished game past TTL should be cleaned up
	game.lastActivity = time.Now().Add(-15 * time.Minute)
	s.cleanupGames()
	
	s.mu.RLock()
	_, stillExists = s.games[gameID]
	s.mu.RUnlock()
	assert.False(t, stillExists, "Old finished game should be cleaned up")
	
	// Test 4: Abandoned game should be cleaned up
	// Create another game
	resp2, err := s.CreateGame(ctx, &gamev1.CreateGameRequest{
		Config: &gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
		},
	})
	require.NoError(t, err)
	gameID2 := resp2.GameId
	
	s.mu.Lock()
	game2 := s.games[gameID2]
	game2.lastActivity = time.Now().Add(-35 * time.Minute)
	s.mu.Unlock()
	
	s.cleanupGames()
	
	s.mu.RLock()
	_, stillExists = s.games[gameID2]
	s.mu.RUnlock()
	assert.False(t, stillExists, "Abandoned game should be cleaned up")
}

func TestLastActivityUpdates(t *testing.T) {
	s := NewServer()
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
	joinResp, err := s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player1",
	})
	require.NoError(t, err)
	
	s.mu.RLock()
	afterJoinActivity := game.lastActivity
	s.mu.RUnlock()
	
	assert.True(t, afterJoinActivity.After(initialActivity), "Join should update last activity")
	
	// Join second player to start the game
	_, err = s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player2",
	})
	require.NoError(t, err)
	
	// Wait a bit
	time.Sleep(10 * time.Millisecond)
	
	// Submit action should update activity
	_, err = s.SubmitAction(ctx, &gamev1.SubmitActionRequest{
		GameId:      gameID,
		PlayerId:    0,
		PlayerToken: joinResp.PlayerToken,
		Action: &gamev1.Action{
			Type:       commonv1.ActionType_ACTION_TYPE_UNSPECIFIED,
			TurnNumber: 0,
		},
	})
	require.NoError(t, err)
	
	s.mu.RLock()
	afterActionActivity := game.lastActivity
	s.mu.RUnlock()
	
	assert.True(t, afterActionActivity.After(afterJoinActivity), "Submit action should update last activity")
}