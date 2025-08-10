package gameserver

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

func TestIdempotencyWithSameKey(t *testing.T) {
	s := NewServer(10)
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

	// Join two players to start the game
	joinResp1, err := s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player1",
	})
	require.NoError(t, err)

	joinResp2, err := s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player2",
	})
	require.NoError(t, err)

	// Submit action with idempotency key - using null action for simplicity
	idempotencyKey := "test-key-123"
	action := &gamev1.Action{
		Type:       commonv1.ActionType_ACTION_TYPE_UNSPECIFIED, // No action this turn
		TurnNumber: 0,
	}

	// First submission
	resp1, err := s.SubmitAction(ctx, &gamev1.SubmitActionRequest{
		GameId:         gameID,
		PlayerId:       0,
		PlayerToken:    joinResp1.PlayerToken,
		Action:         action,
		IdempotencyKey: idempotencyKey,
	})
	require.NoError(t, err)
	assert.True(t, resp1.Success)

	// Second submission with same idempotency key - should return cached response
	resp2, err := s.SubmitAction(ctx, &gamev1.SubmitActionRequest{
		GameId:         gameID,
		PlayerId:       0,
		PlayerToken:    joinResp1.PlayerToken,
		Action:         action,
		IdempotencyKey: idempotencyKey,
	})
	require.NoError(t, err)

	// Should return same response
	assert.Equal(t, resp1.Success, resp2.Success)
	assert.Equal(t, resp1.ErrorCode, resp2.ErrorCode)
	assert.Equal(t, resp1.ErrorMessage, resp2.ErrorMessage)
	assert.Equal(t, resp1.NextTurnNumber, resp2.NextTurnNumber)

	// Submit second player's action to complete the turn
	_, err = s.SubmitAction(ctx, &gamev1.SubmitActionRequest{
		GameId:      gameID,
		PlayerId:    1,
		PlayerToken: joinResp2.PlayerToken,
		Action: &gamev1.Action{
			Type:       commonv1.ActionType_ACTION_TYPE_UNSPECIFIED,
			TurnNumber: 0,
		},
	})
	require.NoError(t, err)

	// Check that game advanced only one turn (not two)
	stateResp, err := s.GetGameState(ctx, &gamev1.GetGameStateRequest{
		GameId:      gameID,
		PlayerId:    0,
		PlayerToken: joinResp1.PlayerToken,
	})
	require.NoError(t, err)
	assert.Equal(t, int32(1), stateResp.State.Turn, "Game should have advanced only one turn")
}

func TestIdempotencyForErrors(t *testing.T) {
	s := NewServer(10)
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

	// Join only one player (game not started)
	joinResp, err := s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player1",
	})
	require.NoError(t, err)

	// Submit action with idempotency key - should fail because game not started
	idempotencyKey := "error-test-key"
	resp1, err := s.SubmitAction(ctx, &gamev1.SubmitActionRequest{
		GameId:         gameID,
		PlayerId:       0,
		PlayerToken:    joinResp.PlayerToken,
		Action:         &gamev1.Action{TurnNumber: 0},
		IdempotencyKey: idempotencyKey,
	})
	require.NoError(t, err)
	assert.False(t, resp1.Success)
	assert.Equal(t, commonv1.ErrorCode_ERROR_CODE_INVALID_PHASE, resp1.ErrorCode)

	// Retry with same idempotency key - should return cached error response
	resp2, err := s.SubmitAction(ctx, &gamev1.SubmitActionRequest{
		GameId:         gameID,
		PlayerId:       0,
		PlayerToken:    joinResp.PlayerToken,
		Action:         &gamev1.Action{TurnNumber: 0},
		IdempotencyKey: idempotencyKey,
	})
	require.NoError(t, err)

	// Should return same error response
	assert.Equal(t, resp1.Success, resp2.Success)
	assert.Equal(t, resp1.ErrorCode, resp2.ErrorCode)
	assert.Equal(t, resp1.ErrorMessage, resp2.ErrorMessage)
}

func TestIdempotencyAcrossPlayers(t *testing.T) {
	s := NewServer(10)
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

	// Join two players to start the game
	joinResp1, err := s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player1",
	})
	require.NoError(t, err)

	joinResp2, err := s.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player2",
	})
	require.NoError(t, err)

	// Both players use the same idempotency key (should be independent)
	idempotencyKey := "shared-key-123"

	// Player 0 submits action
	resp1, err := s.SubmitAction(ctx, &gamev1.SubmitActionRequest{
		GameId:         gameID,
		PlayerId:       0,
		PlayerToken:    joinResp1.PlayerToken,
		Action:         &gamev1.Action{Type: commonv1.ActionType_ACTION_TYPE_UNSPECIFIED, TurnNumber: 0},
		IdempotencyKey: idempotencyKey,
	})
	require.NoError(t, err)
	assert.True(t, resp1.Success)

	// Player 1 submits action with same idempotency key - should succeed
	// because idempotency keys are per-request, not per-game
	resp2, err := s.SubmitAction(ctx, &gamev1.SubmitActionRequest{
		GameId:         gameID,
		PlayerId:       1,
		PlayerToken:    joinResp2.PlayerToken,
		Action:         &gamev1.Action{Type: commonv1.ActionType_ACTION_TYPE_UNSPECIFIED, TurnNumber: 0},
		IdempotencyKey: idempotencyKey,
	})
	require.NoError(t, err)
	assert.True(t, resp2.Success, "Different players should be able to use the same idempotency key")
}
