package gameserver

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/rs/zerolog/log"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	gameengine "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// Server implements the GameService gRPC server
type Server struct {
	gamev1.UnimplementedGameServiceServer
	
	// Game registry for managing active games
	mu      sync.RWMutex
	games   map[string]*gameInstance
	nextID  atomic.Int64
}

type gameInstance struct {
	id      string
	config  *gamev1.GameConfig
	status  commonv1.GameStatus
	players []playerInfo
	engine  *gameengine.Engine
	mu      sync.Mutex // Per-game mutex for thread-safe engine operations
}

type playerInfo struct {
	id    int32
	name  string
	token string
}

// NewServer creates a new game server
func NewServer() *Server {
	return &Server{
		games: make(map[string]*gameInstance),
	}
}

// CreateGame creates a new game instance
func (s *Server) CreateGame(ctx context.Context, req *gamev1.CreateGameRequest) (*gamev1.CreateGameResponse, error) {
	// Generate game ID
	gameID := fmt.Sprintf("game-%d", s.nextID.Add(1))
	
	log.Info().
		Str("game_id", gameID).
		Msg("Creating new game")

	s.mu.Lock()
	defer s.mu.Unlock()

	// Set default config if not provided
	config := req.Config
	if config == nil {
		config = &gamev1.GameConfig{
			Width:       20,
			Height:      20,
			MaxPlayers:  2,
			FogOfWar:    true,
			TurnTimeMs:  0,
		}
	}

	// Create new game instance
	game := &gameInstance{
		id:      gameID,
		config:  config,
		status:  commonv1.GameStatus_GAME_STATUS_WAITING,
		players: make([]playerInfo, 0, config.MaxPlayers),
	}

	s.games[gameID] = game

	return &gamev1.CreateGameResponse{
		GameId: gameID,
		Config: config,
	}, nil
}

// JoinGame allows a player to join an existing game
func (s *Server) JoinGame(ctx context.Context, req *gamev1.JoinGameRequest) (*gamev1.JoinGameResponse, error) {
	log.Info().
		Str("game_id", req.GameId).
		Str("player_name", req.PlayerName).
		Msg("Player joining game")

	s.mu.Lock()
	defer s.mu.Unlock()

	game, exists := s.games[req.GameId]
	if !exists {
		return nil, status.Errorf(codes.NotFound, "game %s not found", req.GameId)
	}

	// Check if player already in game
	for _, p := range game.players {
		if p.name == req.PlayerName {
			// Return existing player info
			return &gamev1.JoinGameResponse{
				PlayerId:    p.id,
				PlayerToken: p.token,
				InitialState: s.createGameState(game, p.id),
			}, nil
		}
	}

	// Check if game is full
	if len(game.players) >= int(game.config.MaxPlayers) {
		return nil, status.Error(codes.ResourceExhausted, "game is full")
	}

	// Add new player
	playerID := int32(len(game.players))
	playerToken := fmt.Sprintf("token-%s-%d", req.GameId, playerID)
	
	game.players = append(game.players, playerInfo{
		id:    playerID,
		name:  req.PlayerName,
		token: playerToken,
	})

	// Start game if we have enough players
	if len(game.players) == int(game.config.MaxPlayers) {
		game.status = commonv1.GameStatus_GAME_STATUS_IN_PROGRESS
		
		// Create the game engine
		engineConfig := gameengine.GameConfig{
			Width:   int(game.config.Width),
			Height:  int(game.config.Height),
			Players: int(game.config.MaxPlayers),
			Logger:  log.Logger,
		}
		game.engine = gameengine.NewEngine(ctx, engineConfig)
		
		if game.engine == nil {
			return nil, status.Error(codes.Internal, "failed to create game engine")
		}
	}

	return &gamev1.JoinGameResponse{
		PlayerId:     playerID,
		PlayerToken:  playerToken,
		InitialState: s.createGameState(game, playerID),
	}, nil
}

// SubmitAction submits a player action to the game
func (s *Server) SubmitAction(ctx context.Context, req *gamev1.SubmitActionRequest) (*gamev1.SubmitActionResponse, error) {
	return nil, status.Error(codes.Unimplemented, "SubmitAction not implemented")
}

// GetGameState retrieves the current game state for a player
func (s *Server) GetGameState(ctx context.Context, req *gamev1.GetGameStateRequest) (*gamev1.GetGameStateResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	game, exists := s.games[req.GameId]
	if !exists {
		return nil, status.Errorf(codes.NotFound, "game %s not found", req.GameId)
	}

	// Validate player token
	valid := false
	for _, p := range game.players {
		if p.id == req.PlayerId && p.token == req.PlayerToken {
			valid = true
			break
		}
	}
	if !valid {
		return nil, status.Error(codes.PermissionDenied, "invalid player credentials")
	}

	return &gamev1.GetGameStateResponse{
		State: s.createGameState(game, req.PlayerId),
	}, nil
}

// StreamGame streams real-time game updates to connected players
func (s *Server) StreamGame(req *gamev1.StreamGameRequest, stream gamev1.GameService_StreamGameServer) error {
	return status.Error(codes.Unimplemented, "StreamGame not implemented")
}

// createGameState creates a GameState message for the given game and player
func (s *Server) createGameState(game *gameInstance, playerID int32) *gamev1.GameState {
	// If engine exists, use it
	if game.engine != nil {
		game.mu.Lock()
		engineState := game.engine.GameState()
		game.mu.Unlock()
		return s.convertGameStateToProto(game, engineState, playerID)
	}
	
	// Otherwise create placeholder state for waiting games
	players := make([]*gamev1.PlayerState, len(game.players))
	for i, p := range game.players {
		players[i] = &gamev1.PlayerState{
			Id:     p.id,
			Name:   p.name,
			Status: commonv1.PlayerStatus_PLAYER_STATUS_ACTIVE,
			ArmyCount: 1,
			TileCount: 1,
			Color:  fmt.Sprintf("#%06X", i*0x333333), // Simple color generation
		}
	}

	// Create empty board
	tiles := make([]*gamev1.Tile, game.config.Width*game.config.Height)
	for i := range tiles {
		tiles[i] = &gamev1.Tile{
			Type:      commonv1.TileType_TILE_TYPE_NORMAL,
			OwnerId:   -1,
			ArmyCount: 0,
			Visible:   true, // For now, make everything visible
			FogOfWar:  false,
		}
	}

	return &gamev1.GameState{
		GameId:   game.id,
		Status:   game.status,
		Turn:     0,
		Board: &gamev1.Board{
			Width:  game.config.Width,
			Height: game.config.Height,
			Tiles:  tiles,
		},
		Players:  players,
		WinnerId: -1,
		ActionMask: make([]bool, 0), // Empty mask for waiting games
	}
}

// GetActiveGames returns the number of active games (for testing)
func (s *Server) GetActiveGames() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.games)
}

// convertTileType converts from core tile type (int) to protobuf TileType
func convertTileType(t int) commonv1.TileType {
	switch t {
	case core.TileNormal:
		return commonv1.TileType_TILE_TYPE_NORMAL
	case core.TileGeneral:
		return commonv1.TileType_TILE_TYPE_GENERAL
	case core.TileCity:
		return commonv1.TileType_TILE_TYPE_CITY
	case core.TileMountain:
		return commonv1.TileType_TILE_TYPE_MOUNTAIN
	default:
		return commonv1.TileType_TILE_TYPE_UNSPECIFIED
	}
}

// convertGameStateToProto converts the game engine state to protobuf format
func (s *Server) convertGameStateToProto(game *gameInstance, engineState gameengine.GameState, playerID int32) *gamev1.GameState {
	// Create player states from engine data
	players := make([]*gamev1.PlayerState, len(engineState.Players))
	for i, p := range engineState.Players {
		playerState := &gamev1.PlayerState{
			Id:        int32(p.ID),
			Name:      game.players[i].name,
			Status:    commonv1.PlayerStatus_PLAYER_STATUS_ACTIVE,
			ArmyCount: int32(p.ArmyCount),
			TileCount: int32(len(p.OwnedTiles)),
			Color:     fmt.Sprintf("#%06X", i*0x333333),
		}
		
		if !p.Alive {
			playerState.Status = commonv1.PlayerStatus_PLAYER_STATUS_ELIMINATED
		}
		
		// Show general position if discovered or eliminated
		if p.GeneralIdx >= 0 && (!p.Alive || engineState.Board.T[p.GeneralIdx].Owner == int(playerID)) {
			x := p.GeneralIdx % engineState.Board.W
			y := p.GeneralIdx / engineState.Board.W
			playerState.GeneralPosition = &commonv1.Coordinate{
				X: int32(x),
				Y: int32(y),
			}
		}
		
		players[i] = playerState
	}
	
	// Convert board tiles with fog of war
	tiles := make([]*gamev1.Tile, len(engineState.Board.T))
	visibility := game.engine.ComputePlayerVisibility(int(playerID))
	
	for i, tile := range engineState.Board.T {
		protoTile := &gamev1.Tile{
			Type:      convertTileType(tile.Type),
			OwnerId:   int32(tile.Owner),
			ArmyCount: int32(tile.Army),
			Visible:   visibility.VisibleTiles[i],
			FogOfWar:  visibility.FogTiles[i],
		}
		
		// Apply fog of war rules
		if !protoTile.Visible && !protoTile.FogOfWar {
			// Completely hidden tile
			protoTile.Type = commonv1.TileType_TILE_TYPE_NORMAL
			protoTile.OwnerId = -1
			protoTile.ArmyCount = 0
		} else if protoTile.FogOfWar && !protoTile.Visible {
			// In fog - show type but not current state
			protoTile.OwnerId = -1
			protoTile.ArmyCount = 0
		}
		
		tiles[i] = protoTile
	}
	
	// Determine winner
	winnerId := int32(-1)
	if game.engine.IsGameOver() {
		winnerId = int32(game.engine.GetWinner())
	}
	
	// Generate action mask for the requesting player
	actionMask := game.engine.GetLegalActionMask(int(playerID))
	
	return &gamev1.GameState{
		GameId:     game.id,
		Status:     game.status,
		Turn:       int32(engineState.Turn),
		Board: &gamev1.Board{
			Width:  int32(engineState.Board.W),
			Height: int32(engineState.Board.H),
			Tiles:  tiles,
		},
		Players:    players,
		WinnerId:   winnerId,
		ActionMask: actionMask,
	}
}