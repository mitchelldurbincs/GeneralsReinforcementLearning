package gameserver

import (
	"context"
	"fmt"
	"time"

	"github.com/rs/zerolog/log"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	gameengine "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	// "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	// "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	// "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/states"
	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// Server implements the GameService gRPC server
type Server struct {
	gamev1.UnimplementedGameServiceServer

	// Game manager for handling all game instances
	gameManager *GameManager

	// Action validator for validating game actions
	validator *ActionValidator

	// Experience service for collecting and streaming experiences
	experienceService *ExperienceService
}

// Server configuration constants
const (
	// Cleanup configuration
	cleanupInterval      = 5 * time.Minute  // How often to run cleanup
	finishedGameTTL      = 10 * time.Minute // Keep finished games for 10 minutes
	abandonedGameTimeout = 30 * time.Minute // Consider game abandoned after 30 minutes of inactivity
)

// NewServer creates a new game server
func NewServer(maxGames int) *Server {
	// Create experience service with its buffer manager
	experienceService := NewExperienceService(nil)

	// Create game manager with experience service
	gameManager := NewGameManagerWithExperience(experienceService, maxGames)

	return &Server{
		gameManager:       gameManager,
		validator:         NewActionValidator(gameManager),
		experienceService: experienceService,
	}
}

// CreateGame creates a new game instance
func (s *Server) CreateGame(ctx context.Context, req *gamev1.CreateGameRequest) (*gamev1.CreateGameResponse, error) {
	// Create new game instance
	game, gameID, err := s.gameManager.CreateGame(req.Config)
	if err != nil {
		return nil, status.Errorf(codes.ResourceExhausted, "failed to create game: %v", err)
	}

	log.Info().
		Str("game_id", gameID).
		Msg("Creating new game")

	return &gamev1.CreateGameResponse{
		GameId: gameID,
		Config: game.config,
	}, nil
}

// JoinGame allows a player to join an existing game
func (s *Server) JoinGame(ctx context.Context, req *gamev1.JoinGameRequest) (*gamev1.JoinGameResponse, error) {
	log.Info().
		Str("game_id", req.GameId).
		Str("player_name", req.PlayerName).
		Msg("Player joining game")

	game, exists := s.gameManager.GetGame(req.GameId)
	if !exists {
		return nil, status.Errorf(codes.NotFound, "game %s not found: request from player %s", req.GameId, req.PlayerName)
	}

	// Lock the game instance for the entire operation
	game.mu.Lock()

	// Check if player already in game
	for _, p := range game.players {
		if p.name == req.PlayerName {
			// Store player info to return after unlocking
			playerId := p.id
			playerToken := p.token
			game.mu.Unlock()
			// Return existing player info
			return &gamev1.JoinGameResponse{
				PlayerId:     playerId,
				PlayerToken:  playerToken,
				InitialState: s.createGameState(game, playerId),
			}, nil
		}
	}

	// Check if game is in a phase that allows joining
	// If engine exists, use its phase. Otherwise, we're still in pre-engine lobby phase
	if game.engine != nil {
		currentPhase := game.currentPhaseUnlocked()
		if currentPhase != commonv1.GamePhase_GAME_PHASE_LOBBY {
			game.mu.Unlock()
			return nil, status.Errorf(codes.FailedPrecondition, "cannot join game %s: game is in %s phase", req.GameId, currentPhase.String())
		}
	}

	// Check if game is full
	if len(game.players) >= int(game.config.MaxPlayers) {
		game.mu.Unlock()
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

	// Update last activity time
	game.lastActivity = time.Now()

	// Start game if we have enough players
	if len(game.players) == int(game.config.MaxPlayers) {
		// Use the game manager's StartEngine method which properly handles experience collection
		// StartEngine already handles:
		// - Creating the engine with experience collector if enabled
		// - Setting up event subscriptions
		// - Initializing action buffer
		engineCtx := context.Background()
		if err := game.StartEngine(engineCtx); err != nil {
			game.mu.Unlock()
			return nil, status.Errorf(codes.Internal, "failed to start game engine for game %s: %v", req.GameId, err)
		}

		log.Info().
			Str("game_id", req.GameId).
			Int32("current_turn", game.currentTurn).
			Bool("action_buffer_initialized", game.actionBuffer != nil).
			Msg("Game state initialized")

		log.Info().
			Str("game_id", req.GameId).
			Int("players", len(game.players)).
			Msg("Game started")

		// Broadcast game started event
		game.streamManager.BroadcastGameStarted()
	}

	// Store values needed after unlock
	shouldStartTimer := len(game.players) == int(game.config.MaxPlayers)
	turnTimeMs := game.config.TurnTimeMs
	
	// Unlock before creating game state to avoid potential deadlock
	game.mu.Unlock()
	
	// Start turn timer after releasing the lock to avoid deadlock
	if shouldStartTimer {
		game.startTurnTimer(game.engineCtx, time.Duration(turnTimeMs)*time.Millisecond, s)
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
		Str("idempotency_key", req.IdempotencyKey).
		Msg("Received action submission")

	// Validate request
	result, game := s.validator.ValidateSubmitActionRequest(ctx, req)

	// Handle cached response
	if result.CachedResponse != nil {
		return result.CachedResponse, nil
	}

	// Handle validation failures
	if !result.Valid {
		resp := &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    result.ErrorCode,
			ErrorMessage: result.ErrorMessage,
		}
		if req.IdempotencyKey != "" && game != nil {
			game.idempotencyManager.Store(req.IdempotencyKey, resp)
		}
		return resp, nil
	}

	// Get current turn for later use
	game.mu.RLock()
	currentTurn := game.currentTurn
	game.mu.RUnlock()

	// Convert protobuf action to core action
	coreAction, err := convertProtoAction(req.Action, req.PlayerId)
	if err != nil {
		resp := &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
			ErrorMessage: fmt.Sprintf("invalid action for game %s player %d: %v", req.GameId, req.PlayerId, err),
		}
		if req.IdempotencyKey != "" {
			game.idempotencyManager.Store(req.IdempotencyKey, resp)
		}
		return resp, nil
	}

	// Validate core action
	actionResult := s.validator.ValidateCoreAction(coreAction, game, req.PlayerId, currentTurn)
	if !actionResult.Valid {
		resp := &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    actionResult.ErrorCode,
			ErrorMessage: actionResult.ErrorMessage,
		}
		if req.IdempotencyKey != "" {
			game.idempotencyManager.Store(req.IdempotencyKey, resp)
		}
		return resp, nil
	}

	// Collect the action and check if all players have submitted
	allSubmitted := game.collectAction(req.PlayerId, coreAction)

	// If all players have submitted, process the turn
	if allSubmitted {
		// Use the engine context to avoid cancellation when the request completes
		if err := game.processTurn(game.engineCtx); err != nil {
			log.Error().Err(err).
				Str("game_id", req.GameId).
				Int32("turn", currentTurn).
				Msg("Failed to process turn")

			resp := &gamev1.SubmitActionResponse{
				Success:      false,
				ErrorCode:    commonv1.ErrorCode_ERROR_CODE_UNSPECIFIED,
				ErrorMessage: fmt.Sprintf("failed to process turn %d for game %s", currentTurn, req.GameId),
			}
			if req.IdempotencyKey != "" {
				game.idempotencyManager.Store(req.IdempotencyKey, resp)
			}
			return resp, nil
		}

		// Broadcast updates to all connected stream clients
		game.broadcastUpdates(s)

		// If game is still active, start next turn timer
		if game.CurrentPhase() == commonv1.GamePhase_GAME_PHASE_RUNNING {
			game.startTurnTimer(game.engineCtx, time.Duration(game.config.TurnTimeMs)*time.Millisecond, s)
		}
	}

	resp := &gamev1.SubmitActionResponse{
		Success:        true,
		NextTurnNumber: currentTurn + 1,
	}
	if req.IdempotencyKey != "" {
		game.idempotencyManager.Store(req.IdempotencyKey, resp)
	}
	return resp, nil
}

// GetGameState retrieves the current game state for a player
func (s *Server) GetGameState(ctx context.Context, req *gamev1.GetGameStateRequest) (*gamev1.GetGameStateResponse, error) {
	game, exists := s.gameManager.GetGame(req.GameId)
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
	log.Info().
		Str("game_id", req.GameId).
		Int32("player_id", req.PlayerId).
		Msg("Player connecting to game stream")

	// Validate game exists
	game, exists := s.gameManager.GetGame(req.GameId)
	if !exists {
		return status.Errorf(codes.NotFound, "game %s not found", req.GameId)
	}

	// Validate player credentials
	authenticated := false
	for _, p := range game.players {
		if p.id == req.PlayerId && p.token == req.PlayerToken {
			authenticated = true
			break
		}
	}

	if !authenticated {
		return status.Errorf(codes.PermissionDenied, "invalid player credentials for game %s: player %d", req.GameId, req.PlayerId)
	}

	// Create stream client with cancellable context
	ctx, cancel := context.WithCancel(stream.Context())
	client := &streamClient{
		playerID:   req.PlayerId,
		stream:     stream,
		ctx:        ctx,
		cancelFunc: cancel,
		updateChan: make(chan *gamev1.GameUpdate, 10), // Buffered channel
	}

	// Register the stream
	game.streamManager.RegisterClient(client)
	defer game.streamManager.UnregisterClient(req.PlayerId)

	// Send initial game state
	initialUpdate := &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_FullState{
			FullState: s.createGameState(game, req.PlayerId),
		},
		Timestamp: timestamppb.Now(),
	}

	if err := stream.Send(initialUpdate); err != nil {
		log.Error().Err(err).
			Str("game_id", req.GameId).
			Int32("player_id", req.PlayerId).
			Msg("Failed to send initial game state")
		return err
	}

	// Start goroutine to handle updates
	errChan := make(chan error, 1)
	go func() {
		for {
			select {
			case update := <-client.updateChan:
				if err := stream.Send(update); err != nil {
					errChan <- err
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	// Wait for stream to close or error
	select {
	case err := <-errChan:
		log.Error().Err(err).
			Str("game_id", req.GameId).
			Int32("player_id", req.PlayerId).
			Msg("Stream error")
		return err
	case <-stream.Context().Done():
		log.Info().
			Str("game_id", req.GameId).
			Int32("player_id", req.PlayerId).
			Msg("Stream closed by client")
		return nil
	}
}

// createGameState creates a GameState message for the given game and player
func (s *Server) createGameState(game *gameInstance, playerID int32) *gamev1.GameState {
	// If engine exists, use it
	if game.engine != nil {
		game.mu.RLock()
		engineState := game.engine.GameState()
		game.mu.RUnlock()
		return s.convertGameStateToProto(game, engineState, playerID)
	}

	// Otherwise create placeholder state for waiting games
	players := make([]*gamev1.PlayerState, len(game.players))
	for i, p := range game.players {
		players[i] = &gamev1.PlayerState{
			Id:        p.id,
			Name:      p.name,
			Status:    commonv1.PlayerStatus_PLAYER_STATUS_ACTIVE,
			ArmyCount: 1,
			TileCount: 1,
			Color:     generatePlayerColor(i), // Simple color generation
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

	// Get current phase (will be UNSPECIFIED if no engine yet)
	currentPhase := game.CurrentPhase()
	if currentPhase == commonv1.GamePhase_GAME_PHASE_UNSPECIFIED {
		// No engine yet, we're in pre-engine lobby
		currentPhase = commonv1.GamePhase_GAME_PHASE_LOBBY
	}

	return &gamev1.GameState{
		GameId: game.id,
		Status: mapPhaseToStatus(currentPhase), // Map phase to status for backward compatibility
		Turn:   0,
		Board: &gamev1.Board{
			Width:  game.config.Width,
			Height: game.config.Height,
			Tiles:  tiles,
		},
		Players:      players,
		WinnerId:     -1,
		ActionMask:   make([]bool, 0), // Empty mask for waiting games
		CurrentPhase: currentPhase,
	}
}

// GetActiveGames returns the number of active games (for testing)
func (s *Server) GetActiveGames() int {
	return s.gameManager.GetActiveGames()
}

// computePlayerVisibilityFromEngine is a helper to get player visibility from the engine
func (s *Server) computePlayerVisibilityFromEngine(engine *gameengine.Engine, playerID int) gameengine.PlayerVisibility {
	return engine.ComputePlayerVisibility(playerID)
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
			Color:     generatePlayerColor(i),
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

	// Get current phase from engine
	currentPhase := game.CurrentPhase()

	return &gamev1.GameState{
		GameId: game.id,
		Status: mapPhaseToStatus(currentPhase), // Map phase to status for backward compatibility
		Turn:   int32(engineState.Turn),
		Board: &gamev1.Board{
			Width:  int32(engineState.Board.W),
			Height: int32(engineState.Board.H),
			Tiles:  tiles,
		},
		Players:      players,
		WinnerId:     winnerId,
		ActionMask:   actionMask,
		CurrentPhase: currentPhase,
	}
}

// broadcastUpdates sends game updates to all connected stream clients
func (g *gameInstance) broadcastUpdates(server *Server) {
	if g.streamManager.GetClientCount() == 0 {
		return // No streams to update
	}

	log.Debug().
		Str("game_id", g.id).
		Int("stream_count", g.streamManager.GetClientCount()).
		Msg("Broadcasting updates to stream clients")

	// Get current engine state
	g.mu.RLock()
	engineState := g.engine.GameState()
	changedTiles := g.engine.GetChangedTiles()
	visibilityChangedTiles := g.engine.GetVisibilityChangedTiles()
	g.mu.RUnlock()

	// Send updates to each connected player
	g.streamManager.ForEachClient(func(playerID int32, client *streamClient) {
		update := g.createStreamUpdate(server, engineState, playerID, changedTiles, visibilityChangedTiles)
		g.streamManager.SendToClient(playerID, update)
	})
}

// createStreamUpdate creates an appropriate update for a player based on changed tiles
func (g *gameInstance) createStreamUpdate(server *Server, engineState gameengine.GameState, playerID int32, changedTiles, visibilityChangedTiles map[int]bool) *gamev1.GameUpdate {
	// If there are few changes, send a delta update; otherwise send full state
	totalChanges := len(changedTiles) + len(visibilityChangedTiles)
	boardSize := engineState.Board.W * engineState.Board.H

	// Use delta updates if less than 20% of the board changed
	if totalChanges > 0 && totalChanges < boardSize/5 {
		// Create delta update
		delta := &gamev1.GameStateDelta{
			Turn:        int32(engineState.Turn),
			TileUpdates: make([]*gamev1.TileUpdate, 0, totalChanges),
		}

		// Get player visibility for applying fog of war
		visibility := server.computePlayerVisibilityFromEngine(g.engine, int(playerID))

		// Add changed tiles
		processedTiles := make(map[int]bool)
		for tileIdx := range changedTiles {
			if processedTiles[tileIdx] {
				continue
			}
			processedTiles[tileIdx] = true

			tile := &engineState.Board.T[tileIdx]
			x := tileIdx % engineState.Board.W
			y := tileIdx / engineState.Board.W

			tileUpdate := &gamev1.TileUpdate{
				Position: &commonv1.Coordinate{X: int32(x), Y: int32(y)},
				Tile: &gamev1.Tile{
					Type:      convertTileType(tile.Type),
					OwnerId:   int32(tile.Owner),
					ArmyCount: int32(tile.Army),
					Visible:   visibility.VisibleTiles[tileIdx],
					FogOfWar:  visibility.FogTiles[tileIdx],
				},
			}

			// Apply fog of war rules
			if !tileUpdate.Tile.Visible && !tileUpdate.Tile.FogOfWar {
				// Completely hidden tile
				tileUpdate.Tile.Type = commonv1.TileType_TILE_TYPE_NORMAL
				tileUpdate.Tile.OwnerId = -1
				tileUpdate.Tile.ArmyCount = 0
			} else if tileUpdate.Tile.FogOfWar && !tileUpdate.Tile.Visible {
				// In fog - show type but not current state
				tileUpdate.Tile.OwnerId = -1
				tileUpdate.Tile.ArmyCount = 0
			}

			delta.TileUpdates = append(delta.TileUpdates, tileUpdate)
		}

		// Add visibility changed tiles that weren't already processed
		for tileIdx := range visibilityChangedTiles {
			if processedTiles[tileIdx] {
				continue
			}

			tile := &engineState.Board.T[tileIdx]
			x := tileIdx % engineState.Board.W
			y := tileIdx / engineState.Board.W

			tileUpdate := &gamev1.TileUpdate{
				Position: &commonv1.Coordinate{X: int32(x), Y: int32(y)},
				Tile: &gamev1.Tile{
					Type:      convertTileType(tile.Type),
					OwnerId:   int32(tile.Owner),
					ArmyCount: int32(tile.Army),
					Visible:   visibility.VisibleTiles[tileIdx],
					FogOfWar:  visibility.FogTiles[tileIdx],
				},
			}

			// Apply fog of war rules
			if !tileUpdate.Tile.Visible && !tileUpdate.Tile.FogOfWar {
				// Completely hidden tile
				tileUpdate.Tile.Type = commonv1.TileType_TILE_TYPE_NORMAL
				tileUpdate.Tile.OwnerId = -1
				tileUpdate.Tile.ArmyCount = 0
			} else if tileUpdate.Tile.FogOfWar && !tileUpdate.Tile.Visible {
				// In fog - show type but not current state
				tileUpdate.Tile.OwnerId = -1
				tileUpdate.Tile.ArmyCount = 0
			}

			delta.TileUpdates = append(delta.TileUpdates, tileUpdate)
		}

		// Add player updates
		delta.PlayerUpdates = make([]*gamev1.PlayerUpdate, 0, len(engineState.Players))
		for i, p := range engineState.Players {
			playerState := &gamev1.PlayerState{
				Id:        int32(p.ID),
				Name:      g.players[i].name,
				Status:    commonv1.PlayerStatus_PLAYER_STATUS_ACTIVE,
				ArmyCount: int32(p.ArmyCount),
				TileCount: int32(len(p.OwnedTiles)),
				Color:     generatePlayerColor(i),
			}

			// Update status if player was eliminated
			if !p.Alive {
				playerState.Status = commonv1.PlayerStatus_PLAYER_STATUS_ELIMINATED

				// Show general position when eliminated
				if p.GeneralIdx >= 0 {
					x := p.GeneralIdx % engineState.Board.W
					y := p.GeneralIdx / engineState.Board.W
					playerState.GeneralPosition = &commonv1.Coordinate{
						X: int32(x),
						Y: int32(y),
					}
				}
			}

			playerUpdate := &gamev1.PlayerUpdate{
				PlayerId: int32(p.ID),
				State:    playerState,
			}
			delta.PlayerUpdates = append(delta.PlayerUpdates, playerUpdate)
		}

		return &gamev1.GameUpdate{
			Update: &gamev1.GameUpdate_Delta{
				Delta: delta,
			},
			Timestamp: timestamppb.Now(),
		}
	}

	// Fall back to full state update for large changes
	return &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_FullState{
			FullState: server.convertGameStateToProto(g, engineState, playerID),
		},
		Timestamp: timestamppb.Now(),
	}
}

// GetExperienceService returns the experience service for integration
func (s *Server) GetExperienceService() *ExperienceService {
	return s.experienceService
}
