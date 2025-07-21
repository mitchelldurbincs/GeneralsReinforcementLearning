package gameserver

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

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
	
	// Action collection for turn-based processing
	actionBuffer map[int32]core.Action // playerID -> action for current turn
	actionMu     sync.Mutex            // Protects actionBuffer
	currentTurn  int32                 // Current turn number
	turnDeadline time.Time             // Deadline for current turn
	turnTimer    *time.Timer           // Timer for turn timeout
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
		return nil, status.Errorf(codes.NotFound, "game %s not found: request from player %s", req.GameId, req.PlayerName)
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
		return nil, status.Errorf(codes.ResourceExhausted, "game %s is full: %d/%d players", req.GameId, len(game.players), game.config.MaxPlayers)
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
			return nil, status.Errorf(codes.Internal, "failed to create game engine for game %s: config %dx%d with %d players", req.GameId, game.config.Width, game.config.Height, game.config.MaxPlayers)
		}
		
		// Initialize action collection
		game.actionBuffer = make(map[int32]core.Action)
		game.currentTurn = 0
		
		// Start turn timer if configured
		if game.config.TurnTimeMs > 0 {
			game.startTurnTimer(ctx, time.Duration(game.config.TurnTimeMs)*time.Millisecond)
		}
		
		log.Info().
			Str("game_id", req.GameId).
			Int("players", len(game.players)).
			Msg("Game started")
	}

	return &gamev1.JoinGameResponse{
		PlayerId:     playerID,
		PlayerToken:  playerToken,
		InitialState: s.createGameState(game, playerID),
	}, nil
}

// SubmitAction submits a player action to the game
func (s *Server) SubmitAction(ctx context.Context, req *gamev1.SubmitActionRequest) (*gamev1.SubmitActionResponse, error) {
	log.Debug().
		Str("game_id", req.GameId).
		Int32("player_id", req.PlayerId).
		Msg("Received action submission")
	
	// Validate game exists
	s.mu.RLock()
	game, exists := s.games[req.GameId]
	s.mu.RUnlock()
	
	if !exists {
		return &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_GAME_NOT_FOUND,
			ErrorMessage: fmt.Sprintf("game %s not found", req.GameId),
		}, nil
	}
	
	// Check game status
	if game.status != commonv1.GameStatus_GAME_STATUS_IN_PROGRESS {
		return &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_GAME_OVER,
			ErrorMessage: fmt.Sprintf("game %s is not in progress (status: %s)", req.GameId, game.status),
		}, nil
	}
	
	// Authenticate player
	authenticated := false
	for _, p := range game.players {
		if p.id == req.PlayerId && p.token == req.PlayerToken {
			authenticated = true
			break
		}
	}
	
	if !authenticated {
		return &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_PLAYER,
			ErrorMessage: fmt.Sprintf("invalid player credentials for game %s: player %d", req.GameId, req.PlayerId),
		}, nil
	}
	
	// Validate turn number
	game.actionMu.Lock()
	currentTurn := game.currentTurn
	game.actionMu.Unlock()
	
	if req.Action != nil && req.Action.TurnNumber != currentTurn {
		return &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
			ErrorMessage: fmt.Sprintf("invalid turn number for game %s: expected %d, got %d", req.GameId, currentTurn, req.Action.TurnNumber),
		}, nil
	}
	
	// Convert protobuf action to core action
	coreAction, err := s.convertProtoAction(req.Action, req.PlayerId)
	if err != nil {
		return &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
			ErrorMessage: fmt.Sprintf("invalid action for game %s player %d: %v", req.GameId, req.PlayerId, err),
		}, nil
	}
	
	// Validate action with game engine if not nil
	if coreAction != nil {
		game.mu.Lock()
		engineState := game.engine.GameState()
		err = coreAction.Validate(engineState.Board, int(req.PlayerId))
		game.mu.Unlock()
		
		if err != nil {
			return &gamev1.SubmitActionResponse{
				Success:      false,
				ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
				ErrorMessage: fmt.Sprintf("action validation failed for game %s player %d turn %d: %v", req.GameId, req.PlayerId, currentTurn, err),
			}, nil
		}
	}
	
	// Collect the action and check if all players have submitted
	allSubmitted := game.collectAction(req.PlayerId, coreAction)
	
	// If all players have submitted, process the turn
	if allSubmitted {
		if err := game.processTurn(ctx); err != nil {
			log.Error().Err(err).
				Str("game_id", req.GameId).
				Int32("turn", currentTurn).
				Msg("Failed to process turn")
			
			return &gamev1.SubmitActionResponse{
				Success:      false,
				ErrorCode:    commonv1.ErrorCode_ERROR_CODE_UNSPECIFIED,
				ErrorMessage: fmt.Sprintf("failed to process turn %d for game %s", currentTurn, req.GameId),
			}, nil
		}
		
		// If turn time is configured and game is still active, start next turn timer
		if game.config.TurnTimeMs > 0 && game.status == commonv1.GameStatus_GAME_STATUS_IN_PROGRESS {
			game.startTurnTimer(ctx, time.Duration(game.config.TurnTimeMs)*time.Millisecond)
		}
	}
	
	return &gamev1.SubmitActionResponse{
		Success:        true,
		NextTurnNumber: currentTurn + 1,
	}, nil
}

// GetGameState retrieves the current game state for a player
func (s *Server) GetGameState(ctx context.Context, req *gamev1.GetGameStateRequest) (*gamev1.GetGameStateResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	game, exists := s.games[req.GameId]
	if !exists {
		return nil, status.Errorf(codes.NotFound, "game %s not found: requested by player %d", req.GameId, req.PlayerId)
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
		return nil, status.Errorf(codes.PermissionDenied, "invalid player credentials for game %s: player %d", req.GameId, req.PlayerId)
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

// Helper methods for action collection and turn processing

// convertProtoAction converts a protobuf action to a core game action
func (s *Server) convertProtoAction(protoAction *gamev1.Action, playerID int32) (core.Action, error) {
	if protoAction == nil {
		return nil, nil // No action this turn
	}
	
	switch protoAction.Type {
	case commonv1.ActionType_ACTION_TYPE_MOVE:
		if protoAction.From == nil || protoAction.To == nil {
			return nil, status.Errorf(codes.InvalidArgument, "move action for player %d requires from and to coordinates", playerID)
		}
		
		return &core.MoveAction{
			PlayerID: int(playerID),
			FromX:    int(protoAction.From.X),
			FromY:    int(protoAction.From.Y),
			ToX:      int(protoAction.To.X),
			ToY:      int(protoAction.To.Y),
			MoveAll:  !protoAction.Half, // In proto, half=true means move half; in core, MoveAll=true means move all
		}, nil
		
	default:
		return nil, status.Errorf(codes.InvalidArgument, "unsupported action type %v for player %d", protoAction.Type, playerID)
	}
}

// collectAction stores an action in the buffer and checks if all players have submitted
func (game *gameInstance) collectAction(playerID int32, action core.Action) bool {
	game.actionMu.Lock()
	defer game.actionMu.Unlock()
	
	// Store the action (nil is valid for no action)
	game.actionBuffer[playerID] = action
	
	// Check if all active players have submitted
	activeCount := 0
	for range game.players {
		// Only count players that are still in the game
		// TODO: Check player status once we track eliminations
		activeCount++
	}
	
	return len(game.actionBuffer) >= activeCount
}

// processTurn executes all collected actions and advances the game state
func (game *gameInstance) processTurn(ctx context.Context) error {
	game.actionMu.Lock()
	
	// Convert map to slice of actions
	actions := make([]core.Action, 0, len(game.actionBuffer))
	for _, action := range game.actionBuffer {
		if action != nil {
			actions = append(actions, action)
		}
	}
	
	// Clear the buffer for next turn
	game.actionBuffer = make(map[int32]core.Action)
	game.currentTurn++
	
	game.actionMu.Unlock()
	
	// Process the turn with the game engine
	game.mu.Lock()
	defer game.mu.Unlock()
	
	err := game.engine.Step(ctx, actions)
	if err != nil {
		return fmt.Errorf("game %s turn %d: failed to process %d actions: %w", game.id, game.currentTurn-1, len(actions), err)
	}
	
	// Check if game has ended
	state := game.engine.GameState()
	// Check how many players are still alive
	aliveCount := 0
	for _, player := range state.Players {
		if player.Alive {
			aliveCount++
		}
	}
	
	// Game ends when only one player remains
	if aliveCount <= 1 {
		game.status = commonv1.GameStatus_GAME_STATUS_FINISHED
		// Cancel any pending turn timer
		if game.turnTimer != nil {
			game.turnTimer.Stop()
		}
	}
	
	return nil
}

// startTurnTimer begins a timer for the current turn
func (game *gameInstance) startTurnTimer(ctx context.Context, duration time.Duration) {
	game.actionMu.Lock()
	game.turnDeadline = time.Now().Add(duration)
	game.actionMu.Unlock()
	
	if game.turnTimer != nil {
		game.turnTimer.Stop()
	}
	
	game.turnTimer = time.AfterFunc(duration, func() {
		// Turn timeout - submit nil actions for players who haven't acted
		game.actionMu.Lock()
		
		// Fill in nil actions for missing players
		for _, player := range game.players {
			if _, exists := game.actionBuffer[player.id]; !exists {
				game.actionBuffer[player.id] = nil
			}
		}
		
		allSubmitted := len(game.actionBuffer) >= len(game.players)
		game.actionMu.Unlock()
		
		if allSubmitted {
			// Process the turn with whatever actions we have
			if err := game.processTurn(ctx); err != nil {
				log.Error().Err(err).
					Str("game_id", game.id).
					Int32("turn", game.currentTurn).
					Msg("Failed to process turn after timeout")
			}
			
			// Start timer for next turn if game is still active
			if game.status == commonv1.GameStatus_GAME_STATUS_IN_PROGRESS && duration > 0 {
				game.startTurnTimer(ctx, duration)
			}
		}
	})
}