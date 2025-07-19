package gameserver

import (
	"context"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

const bufSize = 1024 * 1024

// setupTestServer creates an in-memory gRPC server for testing
func setupTestServer(t *testing.T) (gamev1.GameServiceClient, func()) {
	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	gamev1.RegisterGameServiceServer(s, NewServer())
	
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("Server exited with error: %v", err)
		}
	}()

	conn, err := grpc.DialContext(context.Background(), "bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)

	client := gamev1.NewGameServiceClient(conn)

	cleanup := func() {
		conn.Close()
		s.Stop()
		lis.Close()
	}

	return client, cleanup
}

func TestCreateGame(t *testing.T) {
	client, cleanup := setupTestServer(t)
	defer cleanup()

	ctx := context.Background()

	// Test creating a new game with config
	req := &gamev1.CreateGameRequest{
		Config: &gamev1.GameConfig{
			Width:      20,
			Height:     20,
			MaxPlayers: 4,
			FogOfWar:   true,
		},
	}

	resp, err := client.CreateGame(ctx, req)
	require.NoError(t, err)
	assert.NotEmpty(t, resp.GameId)
	assert.Equal(t, int32(20), resp.Config.Width)
	assert.Equal(t, int32(20), resp.Config.Height)
	assert.Equal(t, int32(4), resp.Config.MaxPlayers)

	// Test creating game without config (should use defaults)
	req2 := &gamev1.CreateGameRequest{}
	resp2, err := client.CreateGame(ctx, req2)
	require.NoError(t, err)
	assert.NotEmpty(t, resp2.GameId)
	assert.NotEqual(t, resp.GameId, resp2.GameId) // Should be different game IDs
	assert.Equal(t, int32(20), resp2.Config.Width) // Default width
	assert.Equal(t, int32(20), resp2.Config.Height) // Default height
	assert.Equal(t, int32(2), resp2.Config.MaxPlayers) // Default max players
}

func TestJoinGame(t *testing.T) {
	client, cleanup := setupTestServer(t)
	defer cleanup()

	ctx := context.Background()

	// Create a game first
	createReq := &gamev1.CreateGameRequest{
		Config: &gamev1.GameConfig{
			Width:      20,
			Height:     20,
			MaxPlayers: 2,
		},
	}
	createResp, err := client.CreateGame(ctx, createReq)
	require.NoError(t, err)
	gameID := createResp.GameId

	// Test joining the game
	joinReq := &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "player-1",
	}
	resp, err := client.JoinGame(ctx, joinReq)
	require.NoError(t, err)
	assert.Equal(t, int32(0), resp.PlayerId)
	assert.NotEmpty(t, resp.PlayerToken)
	assert.NotNil(t, resp.InitialState)
	assert.Equal(t, gameID, resp.InitialState.GameId)
	assert.Equal(t, commonv1.GameStatus_GAME_STATUS_WAITING, resp.InitialState.Status)

	// Test joining again with same player (should return same ID)
	resp2, err := client.JoinGame(ctx, joinReq)
	require.NoError(t, err)
	assert.Equal(t, int32(0), resp2.PlayerId)
	assert.Equal(t, resp.PlayerToken, resp2.PlayerToken)

	// Test joining with different player
	joinReq2 := &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "player-2",
	}
	resp3, err := client.JoinGame(ctx, joinReq2)
	require.NoError(t, err)
	assert.Equal(t, int32(1), resp3.PlayerId)
	assert.NotEmpty(t, resp3.PlayerToken)
	assert.NotEqual(t, resp.PlayerToken, resp3.PlayerToken)
	// Game should start after second player joins
	assert.Equal(t, commonv1.GameStatus_GAME_STATUS_IN_PROGRESS, resp3.InitialState.Status)

	// Test joining full game
	joinReq3 := &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "player-3",
	}
	_, err = client.JoinGame(ctx, joinReq3)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "game is full")

	// Test joining non-existent game
	joinReq4 := &gamev1.JoinGameRequest{
		GameId:     "non-existent",
		PlayerName: "player-4",
	}
	_, err = client.JoinGame(ctx, joinReq4)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestGetGameState(t *testing.T) {
	client, cleanup := setupTestServer(t)
	defer cleanup()

	ctx := context.Background()

	// Create and join a game
	createReq := &gamev1.CreateGameRequest{
		Config: &gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
		},
	}
	createResp, err := client.CreateGame(ctx, createReq)
	require.NoError(t, err)
	gameID := createResp.GameId

	joinResp, err := client.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     gameID,
		PlayerName: "test-player",
	})
	require.NoError(t, err)

	// Get game state with valid credentials
	stateReq := &gamev1.GetGameStateRequest{
		GameId:      gameID,
		PlayerId:    joinResp.PlayerId,
		PlayerToken: joinResp.PlayerToken,
	}
	resp, err := client.GetGameState(ctx, stateReq)
	require.NoError(t, err)
	assert.NotNil(t, resp.State)
	assert.Equal(t, gameID, resp.State.GameId)
	assert.Equal(t, commonv1.GameStatus_GAME_STATUS_WAITING, resp.State.Status)
	assert.Equal(t, int32(10), resp.State.Board.Width)
	assert.Equal(t, int32(10), resp.State.Board.Height)
	assert.Len(t, resp.State.Board.Tiles, 100) // 10x10 board

	// Test with invalid token
	stateReq2 := &gamev1.GetGameStateRequest{
		GameId:      gameID,
		PlayerId:    joinResp.PlayerId,
		PlayerToken: "invalid-token",
	}
	_, err = client.GetGameState(ctx, stateReq2)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid player credentials")

	// Test non-existent game
	stateReq3 := &gamev1.GetGameStateRequest{
		GameId:      "non-existent",
		PlayerId:    0,
		PlayerToken: "some-token",
	}
	_, err = client.GetGameState(ctx, stateReq3)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestProtoMessageSerialization(t *testing.T) {
	// Test creating and serializing various proto messages
	
	// Test Coordinate
	coord := &commonv1.Coordinate{X: 10, Y: 15}
	assert.Equal(t, int32(10), coord.X)
	assert.Equal(t, int32(15), coord.Y)

	// Test PlayerState
	playerState := &gamev1.PlayerState{
		Id:       0,
		Name:     "player-1",
		Status:   commonv1.PlayerStatus_PLAYER_STATUS_ACTIVE,
		ArmyCount: 100,
		TileCount: 25,
		Color:    "#FF0000",
	}
	assert.Equal(t, "player-1", playerState.Name)
	assert.Equal(t, int32(100), playerState.ArmyCount)

	// Test Tile
	tile := &gamev1.Tile{
		Type:      commonv1.TileType_TILE_TYPE_CITY,
		OwnerId:   0,
		ArmyCount: 50,
		Visible:   true,
		FogOfWar:  false,
	}
	assert.Equal(t, commonv1.TileType_TILE_TYPE_CITY, tile.Type)
	assert.Equal(t, int32(50), tile.ArmyCount)

	// Test Action
	action := &gamev1.Action{
		Type: commonv1.ActionType_ACTION_TYPE_MOVE,
		From: &commonv1.Coordinate{X: 5, Y: 5},
		To:   &commonv1.Coordinate{X: 5, Y: 6},
		TurnNumber: 42,
		Half: false,
	}
	assert.Equal(t, commonv1.ActionType_ACTION_TYPE_MOVE, action.Type)
	assert.NotNil(t, action.From)
	assert.Equal(t, int32(5), action.From.X)
	assert.Equal(t, int32(42), action.TurnNumber)

	// Test GameConfig
	config := &gamev1.GameConfig{
		Width:      30,
		Height:     30,
		MaxPlayers: 8,
		FogOfWar:   true,
		TurnTimeMs: 1000,
	}
	assert.Equal(t, int32(30), config.Width)
	assert.Equal(t, int32(8), config.MaxPlayers)
	assert.True(t, config.FogOfWar)
}

func TestServerState(t *testing.T) {
	server := NewServer()
	
	// Initially no games
	assert.Equal(t, 0, server.GetActiveGames())
	
	// Create a game
	ctx := context.Background()
	resp, err := server.CreateGame(ctx, &gamev1.CreateGameRequest{})
	require.NoError(t, err)
	assert.Equal(t, 1, server.GetActiveGames())
	
	// Join the game
	_, err = server.JoinGame(ctx, &gamev1.JoinGameRequest{
		GameId:     resp.GameId,
		PlayerName: "test",
	})
	require.NoError(t, err)
	assert.Equal(t, 1, server.GetActiveGames()) // Still just one game
}