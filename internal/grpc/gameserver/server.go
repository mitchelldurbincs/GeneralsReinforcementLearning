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
	"google.golang.org/protobuf/types/known/timestamppb"

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
	
	// Activity tracking for cleanup
	createdAt    time.Time
	lastActivity time.Time
	
	// Idempotency tracking
	idempotencyCache map[string]*idempotencyEntry
	idempotencyMu    sync.RWMutex
	
	// Stream management
	streamClients   map[int32]*streamClient // playerID -> stream client
	streamClientsMu sync.RWMutex            // Protects streamClients map
}

// streamClient represents a connected stream for a player
type streamClient struct {
	playerID   int32
	stream     gamev1.GameService_StreamGameServer
	ctx        context.Context
	cancelFunc context.CancelFunc
	updateChan chan *gamev1.GameUpdate
}

// idempotencyEntry stores a cached response with timestamp
type idempotencyEntry struct {
	response  *gamev1.SubmitActionResponse
	createdAt time.Time
}

type playerInfo struct {
	id    int32
	name  string
	token string
}

// Server configuration constants
const (
	// Cleanup configuration
	cleanupInterval      = 5 * time.Minute  // How often to run cleanup
	finishedGameTTL      = 10 * time.Minute // Keep finished games for 10 minutes
	abandonedGameTimeout = 30 * time.Minute // Consider game abandoned after 30 minutes of inactivity
)

// NewServer creates a new game server
func NewServer() *Server {
	s := &Server{
		games: make(map[string]*gameInstance),
	}
	
	// Start cleanup goroutine
	go s.runCleanup()
	
	return s
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
	now := time.Now()
	game := &gameInstance{
		id:               gameID,
		config:           config,
		status:           commonv1.GameStatus_GAME_STATUS_WAITING,
		players:          make([]playerInfo, 0, config.MaxPlayers),
		createdAt:        now,
		lastActivity:     now,
		idempotencyCache: make(map[string]*idempotencyEntry),
		streamClients:    make(map[int32]*streamClient),
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
	
	// Update last activity time
	game.lastActivity = time.Now()

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
		
		// Broadcast game started event
		game.broadcastGameStarted()
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
	
	// Check idempotency cache if key provided
	if req.IdempotencyKey != "" {
		if cached := game.checkIdempotency(req.IdempotencyKey); cached != nil {
			log.Debug().
				Str("game_id", req.GameId).
				Str("idempotency_key", req.IdempotencyKey).
				Msg("Returning cached response for idempotent request")
			return cached, nil
		}
	}
	
	// Check game status
	if game.status != commonv1.GameStatus_GAME_STATUS_IN_PROGRESS {
		resp := &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_GAME_OVER,
			ErrorMessage: fmt.Sprintf("game %s is not in progress (status: %s)", req.GameId, game.status),
		}
		if req.IdempotencyKey != "" {
			game.storeIdempotency(req.IdempotencyKey, resp)
		}
		return resp, nil
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
		resp := &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_PLAYER,
			ErrorMessage: fmt.Sprintf("invalid player credentials for game %s: player %d", req.GameId, req.PlayerId),
		}
		if req.IdempotencyKey != "" {
			game.storeIdempotency(req.IdempotencyKey, resp)
		}
		return resp, nil
	}
	
	// Validate turn number
	game.actionMu.Lock()
	currentTurn := game.currentTurn
	game.actionMu.Unlock()
	
	if req.Action != nil && req.Action.TurnNumber != currentTurn {
		resp := &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
			ErrorMessage: fmt.Sprintf("invalid turn number for game %s: expected %d, got %d", req.GameId, currentTurn, req.Action.TurnNumber),
		}
		if req.IdempotencyKey != "" {
			game.storeIdempotency(req.IdempotencyKey, resp)
		}
		return resp, nil
	}
	
	// Convert protobuf action to core action
	coreAction, err := s.convertProtoAction(req.Action, req.PlayerId)
	if err != nil {
		resp := &gamev1.SubmitActionResponse{
			Success:      false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
			ErrorMessage: fmt.Sprintf("invalid action for game %s player %d: %v", req.GameId, req.PlayerId, err),
		}
		if req.IdempotencyKey != "" {
			game.storeIdempotency(req.IdempotencyKey, resp)
		}
		return resp, nil
	}
	
	// Validate action with game engine if not nil
	if coreAction != nil {
		game.mu.Lock()
		engineState := game.engine.GameState()
		err = coreAction.Validate(engineState.Board, int(req.PlayerId))
		game.mu.Unlock()
		
		if err != nil {
			resp := &gamev1.SubmitActionResponse{
				Success:      false,
				ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
				ErrorMessage: fmt.Sprintf("action validation failed for game %s player %d turn %d: %v", req.GameId, req.PlayerId, currentTurn, err),
			}
			if req.IdempotencyKey != "" {
				game.storeIdempotency(req.IdempotencyKey, resp)
			}
			return resp, nil
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
			
			resp := &gamev1.SubmitActionResponse{
				Success:      false,
				ErrorCode:    commonv1.ErrorCode_ERROR_CODE_UNSPECIFIED,
				ErrorMessage: fmt.Sprintf("failed to process turn %d for game %s", currentTurn, req.GameId),
			}
			if req.IdempotencyKey != "" {
				game.storeIdempotency(req.IdempotencyKey, resp)
			}
			return resp, nil
		}
		
		// Broadcast updates to all connected stream clients
		game.broadcastUpdates(s)
		
		// If turn time is configured and game is still active, start next turn timer
		if game.config.TurnTimeMs > 0 && game.status == commonv1.GameStatus_GAME_STATUS_IN_PROGRESS {
			game.startTurnTimer(ctx, time.Duration(game.config.TurnTimeMs)*time.Millisecond)
		}
	}
	
	resp := &gamev1.SubmitActionResponse{
		Success:        true,
		NextTurnNumber: currentTurn + 1,
	}
	if req.IdempotencyKey != "" {
		game.storeIdempotency(req.IdempotencyKey, resp)
	}
	return resp, nil
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
	log.Info().
		Str("game_id", req.GameId).
		Int32("player_id", req.PlayerId).
		Msg("Player connecting to game stream")
		
	// Validate game exists
	s.mu.RLock()
	game, exists := s.games[req.GameId]
	s.mu.RUnlock()
	
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
	game.registerStreamClient(client)
	defer game.unregisterStreamClient(req.PlayerId)
	
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
		
	case commonv1.ActionType_ACTION_TYPE_UNSPECIFIED:
		// No action this turn (wait/skip)
		return nil, nil
		
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
	
	// Update last activity time
	game.lastActivity = time.Now()
	
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
	
	// Track player states before processing
	prevAliveStatus := make(map[int]bool)
	prevState := game.engine.GameState()
	for _, player := range prevState.Players {
		prevAliveStatus[player.ID] = player.Alive
	}
	
	err := game.engine.Step(ctx, actions)
	if err != nil {
		return fmt.Errorf("game %s turn %d: failed to process %d actions: %w", game.id, game.currentTurn-1, len(actions), err)
	}
	
	// Check for player eliminations and game ending
	state := game.engine.GameState()
	aliveCount := 0
	var winnerId int = -1
	
	for _, player := range state.Players {
		if player.Alive {
			aliveCount++
			winnerId = player.ID
		} else if prevAliveStatus[player.ID] && !player.Alive {
			// Player was just eliminated
			game.broadcastPlayerEliminated(int32(player.ID), -1) // TODO: track who eliminated the player
		}
	}
	
	// Game ends when only one player remains
	if aliveCount <= 1 && game.status == commonv1.GameStatus_GAME_STATUS_IN_PROGRESS {
		game.status = commonv1.GameStatus_GAME_STATUS_FINISHED
		// Cancel any pending turn timer
		if game.turnTimer != nil {
			game.turnTimer.Stop()
		}
		
		// Broadcast game ended event
		game.broadcastGameEnded(int32(winnerId))
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
			} else {
				// Broadcast updates to stream clients after timeout-triggered turn
				// Note: We need access to the server instance here
				// For now, we'll skip broadcasting on timeout - this would require refactoring
				// to pass the server instance to the timer function
				log.Debug().
					Str("game_id", game.id).
					Msg("Turn processed after timeout (broadcasting not implemented for timer)")
			}
			
			// Start timer for next turn if game is still active
			if game.status == commonv1.GameStatus_GAME_STATUS_IN_PROGRESS && duration > 0 {
				game.startTurnTimer(ctx, duration)
			}
		}
	})
}

// runCleanup periodically removes finished and abandoned games
func (s *Server) runCleanup() {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()
	
	for range ticker.C {
		s.cleanupGames()
	}
}

// cleanupGames removes finished and abandoned games from memory
func (s *Server) cleanupGames() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	now := time.Now()
	var toDelete []string
	
	for gameID, game := range s.games {
		game.mu.Lock()
		
		// Check if game should be cleaned up
		shouldCleanup := false
		reason := ""
		
		// Remove finished games after TTL
		if game.status == commonv1.GameStatus_GAME_STATUS_FINISHED {
			if now.Sub(game.lastActivity) > finishedGameTTL {
				shouldCleanup = true
				reason = "finished game TTL expired"
			}
		} else if now.Sub(game.lastActivity) > abandonedGameTimeout {
			// Remove abandoned games
			shouldCleanup = true
			reason = "game abandoned (no activity)"
		}
		
		game.mu.Unlock()
		
		if shouldCleanup {
			toDelete = append(toDelete, gameID)
			log.Info().
				Str("game_id", gameID).
				Str("reason", reason).
				Dur("age", now.Sub(game.createdAt)).
				Dur("inactive", now.Sub(game.lastActivity)).
				Msg("Cleaning up game")
		}
	}
	
	// Delete games marked for cleanup
	for _, gameID := range toDelete {
		game := s.games[gameID]
		
		// Cancel any active turn timer
		game.mu.Lock()
		if game.turnTimer != nil {
			game.turnTimer.Stop()
		}
		game.mu.Unlock()
		
		// Close all stream clients
		game.streamClientsMu.Lock()
		for playerID, client := range game.streamClients {
			client.cancelFunc()
			close(client.updateChan)
			delete(game.streamClients, playerID)
		}
		game.streamClientsMu.Unlock()
		
		delete(s.games, gameID)
	}
	
	if len(toDelete) > 0 {
		log.Info().
			Int("cleaned", len(toDelete)).
			Int("remaining", len(s.games)).
			Msg("Game cleanup completed")
	}
}

// checkIdempotency returns a cached response if the idempotency key exists
func (g *gameInstance) checkIdempotency(key string) *gamev1.SubmitActionResponse {
	g.idempotencyMu.RLock()
	defer g.idempotencyMu.RUnlock()
	
	entry, exists := g.idempotencyCache[key]
	if !exists {
		return nil
	}
	
	// Check if entry is still valid (24 hours)
	if time.Since(entry.createdAt) > 24*time.Hour {
		return nil
	}
	
	return entry.response
}

// storeIdempotency caches a response for the given idempotency key
func (g *gameInstance) storeIdempotency(key string, resp *gamev1.SubmitActionResponse) {
	g.idempotencyMu.Lock()
	defer g.idempotencyMu.Unlock()
	
	g.idempotencyCache[key] = &idempotencyEntry{
		response:  resp,
		createdAt: time.Now(),
	}
	
	// Clean up old entries if cache is getting large
	if len(g.idempotencyCache) > 1000 {
		g.cleanupIdempotencyCache()
	}
}

// cleanupIdempotencyCache removes old entries from the cache
// Must be called with idempotencyMu held
func (g *gameInstance) cleanupIdempotencyCache() {
	now := time.Now()
	cutoff := now.Add(-24 * time.Hour)
	
	for key, entry := range g.idempotencyCache {
		if entry.createdAt.Before(cutoff) {
			delete(g.idempotencyCache, key)
		}
	}
}

// registerStreamClient adds a new stream client for a player
func (g *gameInstance) registerStreamClient(client *streamClient) {
	g.streamClientsMu.Lock()
	defer g.streamClientsMu.Unlock()
	
	// Close any existing stream for this player
	if existing, exists := g.streamClients[client.playerID]; exists {
		existing.cancelFunc()
		close(existing.updateChan)
	}
	
	g.streamClients[client.playerID] = client
	
	log.Debug().
		Str("game_id", g.id).
		Int32("player_id", client.playerID).
		Int("total_streams", len(g.streamClients)).
		Msg("Stream client registered")
}

// unregisterStreamClient removes a stream client for a player
func (g *gameInstance) unregisterStreamClient(playerID int32) {
	g.streamClientsMu.Lock()
	defer g.streamClientsMu.Unlock()
	
	if client, exists := g.streamClients[playerID]; exists {
		client.cancelFunc()
		close(client.updateChan)
		delete(g.streamClients, playerID)
		
		log.Debug().
			Str("game_id", g.id).
			Int32("player_id", playerID).
			Int("remaining_streams", len(g.streamClients)).
			Msg("Stream client unregistered")
	}
}

// broadcastUpdates sends game updates to all connected stream clients
func (g *gameInstance) broadcastUpdates(server *Server) {
	g.streamClientsMu.RLock()
	defer g.streamClientsMu.RUnlock()
	
	if len(g.streamClients) == 0 {
		return // No streams to update
	}
	
	log.Debug().
		Str("game_id", g.id).
		Int("stream_count", len(g.streamClients)).
		Msg("Broadcasting updates to stream clients")
	
	// Get current engine state
	g.mu.Lock()
	engineState := g.engine.GameState()
	changedTiles := g.engine.GetChangedTiles()
	visibilityChangedTiles := g.engine.GetVisibilityChangedTiles()
	g.mu.Unlock()
	
	// Send updates to each connected player
	for playerID, client := range g.streamClients {
		update := g.createStreamUpdate(server, engineState, playerID, changedTiles, visibilityChangedTiles)
		
		// Non-blocking send to avoid blocking the game
		select {
		case client.updateChan <- update:
			// Successfully queued update
		default:
			// Channel full, log warning
			log.Warn().
				Str("game_id", g.id).
				Int32("player_id", playerID).
				Msg("Stream update channel full, dropping update")
		}
	}
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
			Turn: int32(engineState.Turn),
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
				Color:     fmt.Sprintf("#%06X", i*0x333333),
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

// broadcastGameStarted sends a game started event to all connected stream clients
func (g *gameInstance) broadcastGameStarted() {
	g.streamClientsMu.RLock()
	defer g.streamClientsMu.RUnlock()
	
	if len(g.streamClients) == 0 {
		return
	}
	
	event := &gamev1.GameEvent{
		Event: &gamev1.GameEvent_GameStarted{
			GameStarted: &gamev1.GameStartedEvent{
				StartedAt: timestamppb.Now(),
			},
		},
	}
	
	update := &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_Event{
			Event: event,
		},
		Timestamp: timestamppb.Now(),
	}
	
	g.sendEventToAllClients(update)
}

// broadcastPlayerEliminated sends a player eliminated event to all connected stream clients
func (g *gameInstance) broadcastPlayerEliminated(playerID int32, eliminatedBy int32) {
	g.streamClientsMu.RLock()
	defer g.streamClientsMu.RUnlock()
	
	if len(g.streamClients) == 0 {
		return
	}
	
	event := &gamev1.GameEvent{
		Event: &gamev1.GameEvent_PlayerEliminated{
			PlayerEliminated: &gamev1.PlayerEliminatedEvent{
				PlayerId:     playerID,
				EliminatedBy: eliminatedBy,
			},
		},
	}
	
	update := &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_Event{
			Event: event,
		},
		Timestamp: timestamppb.Now(),
	}
	
	g.sendEventToAllClients(update)
}

// broadcastGameEnded sends a game ended event to all connected stream clients
func (g *gameInstance) broadcastGameEnded(winnerId int32) {
	g.streamClientsMu.RLock()
	defer g.streamClientsMu.RUnlock()
	
	if len(g.streamClients) == 0 {
		return
	}
	
	event := &gamev1.GameEvent{
		Event: &gamev1.GameEvent_GameEnded{
			GameEnded: &gamev1.GameEndedEvent{
				WinnerId: winnerId,
				EndedAt:  timestamppb.Now(),
			},
		},
	}
	
	update := &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_Event{
			Event: event,
		},
		Timestamp: timestamppb.Now(),
	}
	
	g.sendEventToAllClients(update)
}

// sendEventToAllClients is a helper to send an update to all connected stream clients
func (g *gameInstance) sendEventToAllClients(update *gamev1.GameUpdate) {
	for playerID, client := range g.streamClients {
		select {
		case client.updateChan <- update:
			// Successfully queued event
		default:
			log.Warn().
				Str("game_id", g.id).
				Int32("player_id", playerID).
				Msg("Stream event channel full, dropping event")
		}
	}
}