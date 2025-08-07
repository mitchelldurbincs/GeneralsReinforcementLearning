package gameserver

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/experience"
	gameengine "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/states"
	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

type gameInstance struct {
	id      string
	config  *gamev1.GameConfig
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
	idempotencyManager *IdempotencyManager

	// Stream management
	streamManager *StreamManager

	// Reference to game manager (for accessing experience service)
	gameManager *GameManager

	// Engine context for turn processing
	engineCtx context.Context
}

type playerInfo struct {
	id    int32
	name  string
	token string
}

// GameManager manages all active game instances
type GameManager struct {
	mu                   sync.RWMutex
	games                map[string]*gameInstance
	nextID               int64
	experienceService    *ExperienceService
	experienceAggregator *ExperienceAggregator
}

// NewGameManager creates a new game manager
func NewGameManager() *GameManager {
	gm := &GameManager{
		games: make(map[string]*gameInstance),
	}

	// Start cleanup goroutine
	go gm.runCleanup()

	return gm
}

// NewGameManagerWithExperience creates a new game manager with experience service
func NewGameManagerWithExperience(experienceService *ExperienceService) *GameManager {
	gm := &GameManager{
		games:             make(map[string]*gameInstance),
		experienceService: experienceService,
	}

	// Create and start experience aggregator if experience service is available
	if experienceService != nil {
		gm.experienceAggregator = NewExperienceAggregator(
			experienceService.GetBufferManager(),
			log.Logger,
		)
		gm.experienceAggregator.Start()
		log.Info().Msg("Started centralized experience aggregator")
	}

	// Start cleanup goroutine
	go gm.runCleanup()

	return gm
}

// CreateGame creates a new game instance
func (gm *GameManager) CreateGame(config *gamev1.GameConfig) (*gameInstance, string) {
	// Generate game ID
	gm.mu.Lock()
	gm.nextID++
	gameID := fmt.Sprintf("game-%d", gm.nextID)
	gm.mu.Unlock()

	// Set default config if not provided
	if config == nil {
		config = &gamev1.GameConfig{
			Width:      20,
			Height:     20,
			MaxPlayers: 2,
			FogOfWar:   true,
			TurnTimeMs: 0,
		}
	}

	// Create new game instance
	now := time.Now()
	game := &gameInstance{
		id:                 gameID,
		config:             config,
		players:            make([]playerInfo, 0, config.MaxPlayers),
		createdAt:          now,
		lastActivity:       now,
		idempotencyManager: NewIdempotencyManager(),
		streamManager:      NewStreamManager(),
		gameManager:        gm,
	}

	gm.mu.Lock()
	gm.games[gameID] = game
	gm.mu.Unlock()

	return game, gameID
}

// GetGame retrieves a game by ID
func (gm *GameManager) GetGame(gameID string) (*gameInstance, bool) {
	gm.mu.RLock()
	defer gm.mu.RUnlock()
	game, exists := gm.games[gameID]
	return game, exists
}

// GetActiveGames returns the number of active games
func (gm *GameManager) GetActiveGames() int {
	gm.mu.RLock()
	defer gm.mu.RUnlock()
	return len(gm.games)
}

// runCleanup periodically removes finished and abandoned games
func (gm *GameManager) runCleanup() {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		gm.cleanupGames()
	}
}

// cleanupGames removes finished and abandoned games from memory
func (gm *GameManager) cleanupGames() {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	now := time.Now()
	var toDelete []string

	for gameID, game := range gm.games {
		game.mu.Lock()

		// Check if game should be cleaned up
		shouldCleanup := false
		reason := ""

		// Remove finished games after TTL
		currentPhase := game.currentPhaseUnlocked()
		if currentPhase == commonv1.GamePhase_GAME_PHASE_ENDED {
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
		game := gm.games[gameID]

		// Cancel any active turn timer
		game.mu.Lock()
		if game.turnTimer != nil {
			game.turnTimer.Stop()
		}
		game.mu.Unlock()

		// Unregister from experience aggregator
		if gm.experienceAggregator != nil {
			gm.experienceAggregator.UnregisterGame(gameID)
		}

		// Close all stream clients
		game.streamManager.CloseAll()

		delete(gm.games, gameID)
	}

	if len(toDelete) > 0 {
		log.Info().
			Int("cleaned", len(toDelete)).
			Int("remaining", len(gm.games)).
			Msg("Game cleanup completed")
	}
}

// Game instance methods

// AddPlayer adds a new player to the game
func (g *gameInstance) AddPlayer(playerName string) (int32, string, bool) {
	// Check if player already in game
	for _, p := range g.players {
		if p.name == playerName {
			// Return existing player info
			return p.id, p.token, true
		}
	}

	// Check if game is full
	if len(g.players) >= int(g.config.MaxPlayers) {
		return 0, "", false
	}

	// Add new player
	playerID := int32(len(g.players))
	playerToken := fmt.Sprintf("token-%s-%d", g.id, playerID)

	g.players = append(g.players, playerInfo{
		id:    playerID,
		name:  playerName,
		token: playerToken,
	})

	// Update last activity time
	g.lastActivity = time.Now()

	return playerID, playerToken, true
}

// IsPlayerValid validates player credentials
func (g *gameInstance) IsPlayerValid(playerID int32, token string) bool {
	for _, p := range g.players {
		if p.id == playerID && p.token == token {
			return true
		}
	}
	return false
}

// IsFull returns true if the game has reached max players
func (g *gameInstance) IsFull() bool {
	return len(g.players) == int(g.config.MaxPlayers)
}

// getGameManager returns the game manager reference
func (g *gameInstance) getGameManager() *GameManager {
	return g.gameManager
}

// StartEngine initializes the game engine
func (g *gameInstance) StartEngine(ctx context.Context) error {
	if g.engine != nil {
		return fmt.Errorf("engine already started")
	}

	// Create experience collector if experience service is available AND collection is enabled
	var experienceCollector gameengine.ExperienceCollector
	if gm := g.getGameManager(); gm != nil && gm.experienceService != nil && g.config.CollectExperiences {
		// Create a simple collector that writes to the experience buffer
		simpleCollector := experience.NewSimpleCollector(10000, g.id, log.Logger)
		experienceCollector = simpleCollector

		// Register with the centralized aggregator instead of creating a per-game goroutine
		if gm.experienceAggregator != nil {
			gm.experienceAggregator.RegisterGame(g.id, simpleCollector)
			log.Debug().
				Str("game_id", g.id).
				Msg("Registered game with experience aggregator")
		} else {
			// Fallback to old behavior if aggregator not available
			buffer := gm.experienceService.GetBufferManager().GetOrCreateBuffer(g.id)

			// Connect collector to buffer
			go func() {
				// This goroutine transfers experiences from collector to buffer
				ticker := time.NewTicker(100 * time.Millisecond)
				defer ticker.Stop()

				for {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:
						// Get experiences from collector
						if collector, ok := experienceCollector.(*experience.SimpleCollector); ok {
							experiences := collector.GetExperiences()
							if len(experiences) > 0 {
								// Add to buffer
								for _, exp := range experiences {
									if err := buffer.Add(exp); err != nil {
										log.Error().Err(err).Msg("Failed to add experience to buffer")
									}
								}
								// Clear collector after transfer
								collector.Clear()
							}
						}
					}
				}
			}()
		}
	}

	// Create the game engine
	engineConfig := gameengine.GameConfig{
		Width:               int(g.config.Width),
		Height:              int(g.config.Height),
		Players:             int(g.config.MaxPlayers),
		Logger:              log.Logger,
		GameID:              g.id,
		ExperienceCollector: experienceCollector,
	}
	// Store the engine context for use in turn processing
	g.engineCtx = ctx

	g.engine = gameengine.NewEngine(ctx, engineConfig)

	if g.engine == nil {
		return fmt.Errorf("failed to create game engine")
	}

	// Subscribe to state transition events from the engine
	g.engine.EventBus().SubscribeFunc("state.transition", func(event events.Event) {
		// Type assert to StateTransitionEvent
		if stateEvent, ok := event.(*events.StateTransitionEvent); ok {
			// Convert internal phases to proto phases
			previousPhase := convertPhaseToProto(states.GamePhase(0))
			newPhase := convertPhaseToProto(states.GamePhase(0))

			// Parse phase names to get the actual phases
			for i := 0; i <= int(states.PhaseReset); i++ {
				phase := states.GamePhase(i)
				if phase.String() == stateEvent.FromPhase {
					previousPhase = convertPhaseToProto(phase)
				}
				if phase.String() == stateEvent.ToPhase {
					newPhase = convertPhaseToProto(phase)
				}
			}

			// Broadcast the phase change to all stream clients
			g.streamManager.BroadcastPhaseChanged(previousPhase, newPhase, stateEvent.Reason)
		}
	})

	// Initialize action collection
	g.actionBuffer = make(map[int32]core.Action)
	g.currentTurn = 0

	return nil
}

// CurrentPhase returns the current game phase from the engine's state machine
func (g *gameInstance) CurrentPhase() commonv1.GamePhase {
	g.mu.Lock()
	defer g.mu.Unlock()

	return g.currentPhaseUnlocked()
}

// currentPhaseUnlocked returns the current game phase without locking (caller must hold lock)
func (g *gameInstance) currentPhaseUnlocked() commonv1.GamePhase {
	if g.engine == nil {
		return commonv1.GamePhase_GAME_PHASE_UNSPECIFIED
	}

	// Get the current phase from the engine
	enginePhase := g.engine.CurrentPhase()

	// Convert internal phase to proto phase
	return convertPhaseToProto(enginePhase)
}

// collectAction stores an action in the buffer and checks if all players have submitted
func (g *gameInstance) collectAction(playerID int32, action core.Action) bool {
	g.actionMu.Lock()
	defer g.actionMu.Unlock()

	// Store the action (nil is valid for no action)
	g.actionBuffer[playerID] = action

	// Update last activity time
	g.lastActivity = time.Now()

	// Check if all active players have submitted
	activeCount := 0
	for range g.players {
		// Only count players that are still in the game
		// TODO: Check player status once we track eliminations
		activeCount++
	}

	return len(g.actionBuffer) >= activeCount
}

// processTurn executes all collected actions and advances the game state
func (g *gameInstance) processTurn(ctx context.Context) error {
	g.actionMu.Lock()

	// Convert map to slice of actions
	actions := make([]core.Action, 0, len(g.actionBuffer))
	for _, action := range g.actionBuffer {
		if action != nil {
			actions = append(actions, action)
		}
	}

	// Clear the buffer for next turn
	g.actionBuffer = make(map[int32]core.Action)

	g.actionMu.Unlock()

	// Process the turn with the game engine
	g.mu.Lock()
	defer g.mu.Unlock()

	// Track player states before processing
	prevAliveStatus := make(map[int]bool)
	prevState := g.engine.GameState()
	for _, player := range prevState.Players {
		prevAliveStatus[player.ID] = player.Alive
	}

	// Get the current turn from the engine before processing
	currentTurn := g.engine.GameState().Turn

	// Use the engine context instead of the request context to avoid cancellation
	err := g.engine.Step(g.engineCtx, actions)
	if err != nil {
		return fmt.Errorf("game %s turn %d: failed to process %d actions: %w", g.id, currentTurn, len(actions), err)
	}

	// Update our turn counter to match the engine's
	g.currentTurn = int32(g.engine.GameState().Turn)

	// Notify experience aggregator if experiences were collected
	if gm := g.getGameManager(); gm != nil && gm.experienceAggregator != nil && g.config.CollectExperiences {
		gm.experienceAggregator.NotifyExperienceAvailable(g.id)
	}

	// Check for player eliminations and game ending
	state := g.engine.GameState()
	aliveCount := 0
	var winnerId int = -1

	for _, player := range state.Players {
		if player.Alive {
			aliveCount++
			winnerId = player.ID
		} else if prevAliveStatus[player.ID] && !player.Alive {
			// Player was just eliminated
			g.streamManager.BroadcastPlayerEliminated(int32(player.ID), -1) // TODO: track who eliminated the player
		}
	}

	// Game ends when only one player remains
	// The engine will handle the state transition to Ending/Ended
	if aliveCount <= 1 && g.engine != nil {
		currentPhase := g.CurrentPhase()
		if currentPhase == commonv1.GamePhase_GAME_PHASE_RUNNING {
			// The engine's checkGameOver will transition to Ending/Ended
			// We just need to cancel the turn timer
			if g.turnTimer != nil {
				g.turnTimer.Stop()
			}
		}

		// Broadcast game ended event
		g.streamManager.BroadcastGameEnded(int32(winnerId))
	}

	return nil
}

// startTurnTimer begins a timer for the current turn
func (g *gameInstance) startTurnTimer(ctx context.Context, duration time.Duration, server *Server) {
	g.actionMu.Lock()
	g.turnDeadline = time.Now().Add(duration)
	g.actionMu.Unlock()

	if g.turnTimer != nil {
		g.turnTimer.Stop()
	}

	// For 0 duration, process after a short delay to allow action submission
	if duration == 0 {
		go func() {
			// Give players more time to submit actions
			time.Sleep(500 * time.Millisecond)
			g.processTurnTimeout(ctx, duration, server)
		}()
		return
	}

	g.turnTimer = time.AfterFunc(duration, func() {
		g.processTurnTimeout(ctx, duration, server)
	})
}

// processTurnTimeout handles turn timeout logic
func (g *gameInstance) processTurnTimeout(ctx context.Context, duration time.Duration, server *Server) {
	// Turn timeout - submit nil actions for players who haven't acted
	g.actionMu.Lock()

	// Fill in nil actions for missing players
	for _, player := range g.players {
		if _, exists := g.actionBuffer[player.id]; !exists {
			g.actionBuffer[player.id] = nil
		}
	}

	allSubmitted := len(g.actionBuffer) >= len(g.players)
	g.actionMu.Unlock()

	if allSubmitted {
		// Process the turn with whatever actions we have
		// Use the engine context to avoid cancellation issues
		if err := g.processTurn(g.engineCtx); err != nil {
			log.Error().Err(err).
				Str("game_id", g.id).
				Int32("turn", g.currentTurn).
				Msg("Failed to process turn after timeout")
		} else {
			log.Debug().
				Str("game_id", g.id).
				Int32("turn", g.currentTurn).
				Msg("Turn processed after timeout")

			// Broadcast updates to all connected stream clients
			if server != nil {
				g.broadcastUpdates(server)
			}
		}

		// Start timer for next turn if game is still active
		if g.CurrentPhase() == commonv1.GamePhase_GAME_PHASE_RUNNING {
			g.startTurnTimer(g.engineCtx, duration, server)
		}
	}
}
