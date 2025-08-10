package gameserver

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// mockStreamServer implements the GameService_StreamGameServer interface for testing
type mockStreamServer struct {
	grpc.ServerStream
	ctx     context.Context
	mu      sync.RWMutex
	updates []*gamev1.GameUpdate
}

func (m *mockStreamServer) Send(update *gamev1.GameUpdate) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.updates = append(m.updates, update)
	return nil
}

func (m *mockStreamServer) Context() context.Context {
	return m.ctx
}

func (m *mockStreamServer) getUpdates() []*gamev1.GameUpdate {
	m.mu.RLock()
	defer m.mu.RUnlock()
	// Return a copy to avoid races
	result := make([]*gamev1.GameUpdate, len(m.updates))
	copy(result, m.updates)
	return result
}

func TestStreamGame(t *testing.T) {
	// Create server
	server := NewServer(10)

	// Create a game
	createResp, err := server.CreateGame(context.Background(), &gamev1.CreateGameRequest{
		Config: &gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
			FogOfWar:   true,
		},
	})
	require.NoError(t, err)
	gameID := createResp.GameId

	// Join as player 1
	joinResp1, err := server.JoinGame(context.Background(), &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player1",
	})
	require.NoError(t, err)

	// Create mock stream for player 1
	ctx1, cancel1 := context.WithCancel(context.Background())
	defer cancel1()
	stream1 := &mockStreamServer{ctx: ctx1}

	// Start streaming in a goroutine
	streamErr := make(chan error, 1)
	go func() {
		err := server.StreamGame(&gamev1.StreamGameRequest{
			GameId:      gameID,
			PlayerId:    joinResp1.PlayerId,
			PlayerToken: joinResp1.PlayerToken,
		}, stream1)
		streamErr <- err
	}()

	// Give stream time to connect
	time.Sleep(50 * time.Millisecond)

	// Verify initial state was sent
	updates := stream1.getUpdates()
	assert.Len(t, updates, 1)
	assert.NotNil(t, updates[0].GetFullState())

	// Join as player 2 to start the game
	joinResp2, err := server.JoinGame(context.Background(), &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "Player2",
	})
	require.NoError(t, err)

	// Give time for event to propagate
	time.Sleep(50 * time.Millisecond)

	// Verify game started event was sent
	updates = stream1.getUpdates()
	assert.Greater(t, len(updates), 1)
	foundStartEvent := false
	for _, update := range updates[1:] {
		if event := update.GetEvent(); event != nil {
			if event.GetGameStarted() != nil {
				foundStartEvent = true
				break
			}
		}
	}
	assert.True(t, foundStartEvent, "Expected to find game started event")

	// Submit actions from both players
	_, err = server.SubmitAction(context.Background(), &gamev1.SubmitActionRequest{
		GameId:      gameID,
		PlayerId:    joinResp1.PlayerId,
		PlayerToken: joinResp1.PlayerToken,
		Action:      nil, // No action this turn
	})
	require.NoError(t, err)

	_, err = server.SubmitAction(context.Background(), &gamev1.SubmitActionRequest{
		GameId:      gameID,
		PlayerId:    joinResp2.PlayerId,
		PlayerToken: joinResp2.PlayerToken,
		Action:      nil, // No action this turn
	})
	require.NoError(t, err)

	// Give time for updates to propagate
	time.Sleep(50 * time.Millisecond)

	// Verify we received a game update (either full state or delta)
	updates = stream1.getUpdates()
	lastUpdate := updates[len(updates)-1]
	assert.True(t, lastUpdate.GetFullState() != nil || lastUpdate.GetDelta() != nil,
		"Expected to receive either full state or delta update after turn processing")

	// Cancel stream
	cancel1()

	// Wait for stream to finish
	select {
	case <-streamErr:
		// Stream finished
	case <-time.After(time.Second):
		t.Fatal("Stream did not finish within timeout")
	}
}

func TestStreamGameInvalidCredentials(t *testing.T) {
	server := NewServer(10)

	// Create a game
	createResp, err := server.CreateGame(context.Background(), &gamev1.CreateGameRequest{})
	require.NoError(t, err)

	// Try to stream with invalid credentials
	stream := &mockStreamServer{ctx: context.Background()}
	err = server.StreamGame(&gamev1.StreamGameRequest{
		GameId:      createResp.GameId,
		PlayerId:    999,
		PlayerToken: "invalid-token",
	}, stream)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid player credentials")
}

func TestStreamGameNonExistentGame(t *testing.T) {
	server := NewServer(10)

	stream := &mockStreamServer{ctx: context.Background()}
	err := server.StreamGame(&gamev1.StreamGameRequest{
		GameId:      "non-existent-game",
		PlayerId:    0,
		PlayerToken: "token",
	}, stream)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}
