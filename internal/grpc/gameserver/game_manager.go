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
	mu      sync.RWMutex // Single mutex for all game state (upgraded to RWMutex)

	// State management
	stateMachine *states.StateMachine // State machine for game lifecycle
	eventBus     *events.EventBus     // Event bus for state transitions and game events

	// Action collection for turn-based processing
	actionBuffer map[int32]core.Action // playerID -> action for current turn
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
	maxGames             int
	experienceService    *ExperienceService
	experienceAggregator *ExperienceAggregator
}

// NewGameManager creates a new game manager
func NewGameManager(maxGames int) *GameManager {
	gm := &GameManager{
		games:    make(map[string]*gameInstance),
		maxGames: maxGames,
	}

	// Start cleanup goroutine
	go gm.runCleanup()

	return gm
}

// NewGameManagerWithExperience creates a new game manager with experience service
func NewGameManagerWithExperience(experienceService *ExperienceService, maxGames int) *GameManager {
	gm := &GameManager{
		games:             make(map[string]*gameInstance),
		maxGames:          maxGames,
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
func (gm *GameManager) CreateGame(config *gamev1.GameConfig) (*gameInstance, string, error) {
	// Check if we're at the max games limit
	gm.mu.RLock()
	currentGames := len(gm.games)
	maxGames := gm.maxGames
	gm.mu.RUnlock()

	if maxGames > 0 && currentGames >= maxGames {
		log.Warn().
			Int("current_games", currentGames).
			Int("max_games", maxGames).
			Msg("Rejecting game creation - server at capacity")
		return nil, "", fmt.Errorf("server at capacity: %d/%d games active", currentGames, maxGames)
	}

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

	// Create event bus for this game
	eventBus := events.NewEventBus()

	// Create game context for state machine
	gameContext := states.NewGameContext(gameID, int(config.MaxPlayers), log.Logger)

	// Create state machine starting in Initializing phase
	stateMachine := states.NewStateMachine(gameContext, eventBus)

	// Create new game instance
	now := time.Now()
	game := &gameInstance{
		id:                 gameID,
		config:             config,
		players:            make([]playerInfo, 0, config.MaxPlayers),
		stateMachine:       stateMachine,
		eventBus:           eventBus,
		createdAt:          now,
		lastActivity:       now,
		idempotencyManager: NewIdempotencyManager(),
		streamManager:      NewStreamManager(),
		gameManager:        gm,
	}

	// Transition from Initializing to Lobby phase
	if err := stateMachine.TransitionTo(states.PhaseLobby, "Game created and waiting for players"); err != nil {
		return nil, "", fmt.Errorf("failed to transition to lobby phase: %w", err)
	}

	// Subscribe to state transition events to broadcast to stream clients
	eventBus.SubscribeFunc("state.transition", func(event events.Event) {
		if e, ok := event.(*events.StateTransitionEvent); ok {
			// Broadcast phase change to all stream clients
			log.Debug().
				Str("game_id", gameID).
				Str("from_phase", e.FromPhase).
				Str("to_phase", e.ToPhase).
				Str("reason", e.Reason).
				Msg("State transition occurred")

			// Parse the phase strings to states.GamePhase
			fromPhase := states.ParsePhase(e.FromPhase)
			toPhase := states.ParsePhase(e.ToPhase)

			// Notify stream manager of phase change
			if game.streamManager != nil {
				game.streamManager.BroadcastPhaseChanged(
					convertPhaseToProto(fromPhase),
					convertPhaseToProto(toPhase),
					e.Reason,
				)
			}
		}
	})

	// Log experience collection configuration and state
	log.Info().
		Str("game_id", gameID).
		Bool("collect_experiences", config.CollectExperiences).
		Str("initial_phase", stateMachine.CurrentPhase().String()).
		Msg("Created game instance with state machine")

	gm.mu.Lock()
	gm.games[gameID] = game
	currentCount := len(gm.games)
	gm.mu.Unlock()

	log.Info().
		Str("game_id", gameID).
		Int("current_games", currentCount).
		Int("max_games", maxGames).
		Int32("width", config.Width).
		Int32("height", config.Height).
		Int32("max_players", config.MaxPlayers).
		Msg("Successfully created new game")

	return game, gameID, nil
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
	defer func() {
		if r := recover(); r != nil {
			log.Error().
				Interface("panic", r).
				Msg("Game cleanup goroutine panicked - restarting")
			// Restart the cleanup goroutine after a panic
			time.Sleep(5 * time.Second)
			go gm.runCleanup()
		}
	}()

	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		gm.cleanupGames()
	}
}

// cleanupGames removes finished and abandoned games from memory
func (gm *GameManager) cleanupGames() {
	// Phase 1: Collect game references without holding GameManager lock while accessing game locks
	gm.mu.RLock()
	gameRefs := make([]*gameInstance, 0, len(gm.games))
	gameIDs := make([]string, 0, len(gm.games))
	for gameID, game := range gm.games {
		gameRefs = append(gameRefs, game)
		gameIDs = append(gameIDs, gameID)
	}
	gm.mu.RUnlock()

	// Phase 2: Check each game independently (no nested locks)
	now := time.Now()
	var toDelete []string

	for i, game := range gameRefs {
		gameID := gameIDs[i]

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

		// Store these for logging outside the lock
		createdAt := game.createdAt
		lastActivity := game.lastActivity

		game.mu.Unlock()

		if shouldCleanup {
			toDelete = append(toDelete, gameID)
			log.Info().
				Str("game_id", gameID).
				Str("reason", reason).
				Dur("age", now.Sub(createdAt)).
				Dur("inactive", now.Sub(lastActivity)).
				Msg("Cleaning up game")
		}
	}

	// Phase 3: Clean up games with fresh locks (no nesting)
	if len(toDelete) > 0 {
		// First, handle per-game cleanup without holding GameManager lock
		for _, gameID := range toDelete {
			// Get game reference
			gm.mu.RLock()
			game, exists := gm.games[gameID]
			gm.mu.RUnlock()

			if !exists {
				continue
			}

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
		}

		// Finally, remove games from the map with a single lock
		gm.mu.Lock()
		for _, gameID := range toDelete {
			delete(gm.games, gameID)
		}
		remainingCount := len(gm.games)
		gm.mu.Unlock()

		log.Info().
			Int("cleaned", len(toDelete)).
			Int("remaining", remainingCount).
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

	// Update state machine's game context with new player count
	if g.stateMachine != nil && g.stateMachine.GetContext() != nil {
		g.stateMachine.GetContext().PlayerCount = len(g.players)
	}

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

	log.Debug().
		Str("game_id", g.id).
		Bool("collect_experiences", g.config.CollectExperiences).
		Bool("has_game_manager", g.getGameManager() != nil).
		Bool("has_experience_service", g.getGameManager() != nil && g.getGameManager().experienceService != nil).
		Msg("Checking experience collection setup")

	if gm := g.getGameManager(); gm != nil && gm.experienceService != nil && g.config.CollectExperiences {
		// Create a simple collector that writes to the experience buffer
		simpleCollector := experience.NewSimpleCollector(10000, g.id, log.Logger)
		experienceCollector = simpleCollector

		log.Info().
			Str("game_id", g.id).
			Msg("Created experience collector for game")

		// Register with the centralized aggregator instead of creating a per-game goroutine
		if gm.experienceAggregator != nil {
			gm.experienceAggregator.RegisterGame(g.id, simpleCollector)
			log.Info().
				Str("game_id", g.id).
				Msg("Registered game with experience aggregator")
		} else {
			// Fallback to old behavior if aggregator not available
			buffer := gm.experienceService.GetBufferManager().GetOrCreateBuffer(g.id)

			// Connect collector to buffer
			go func() {
				defer func() {
					if r := recover(); r != nil {
						log.Error().
							Interface("panic", r).
							Str("game_id", g.id).
							Msg("Experience collection goroutine panicked")
					}
				}()

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

	log.Info().
		Str("game_id", g.id).
		Bool("has_experience_collector", experienceCollector != nil).
		Msg("Creating game engine with configuration")

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

// GetStateMachine returns the game's state machine
func (g *gameInstance) GetStateMachine() *states.StateMachine {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.stateMachine
}

// CurrentPhase returns the current game phase from the state machine
func (g *gameInstance) CurrentPhase() commonv1.GamePhase {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return g.currentPhaseUnlocked()
}

// currentPhaseUnlocked returns the current game phase without locking (caller must hold lock)
func (g *gameInstance) currentPhaseUnlocked() commonv1.GamePhase {
	// Use state machine directly if available
	if g.stateMachine != nil {
		return convertPhaseToProto(g.stateMachine.CurrentPhase())
	}

	// Fallback to engine for backward compatibility (will be removed)
	if g.engine != nil {
		enginePhase := g.engine.CurrentPhase()
		return convertPhaseToProto(enginePhase)
	}

	return commonv1.GamePhase_GAME_PHASE_UNSPECIFIED
}

// collectAction stores an action in the buffer and checks if all players have submitted
func (g *gameInstance) collectAction(playerID int32, action core.Action) bool {
	g.mu.Lock()
	defer g.mu.Unlock()

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
	g.mu.Lock()
	defer g.mu.Unlock()

	// Convert map to slice of actions
	actions := make([]core.Action, 0, len(g.actionBuffer))
	for _, action := range g.actionBuffer {
		if action != nil {
			actions = append(actions, action)
		}
	}

	// Clear the buffer for next turn
	g.actionBuffer = make(map[int32]core.Action)

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
	g.mu.Lock()
	g.turnDeadline = time.Now().Add(duration)
	g.mu.Unlock()

	if g.turnTimer != nil {
		g.turnTimer.Stop()
	}

	// For 0 duration, process after a short delay to allow action submission
	if duration == 0 {
		go func() {
			defer func() {
				if r := recover(); r != nil {
					log.Error().
						Interface("panic", r).
						Str("game_id", g.id).
						Msg("Turn timeout goroutine panicked")
				}
			}()

			// Give players more time to submit actions
			time.Sleep(500 * time.Millisecond)
			g.processTurnTimeout(ctx, duration, server)
		}()
		return
	}

	g.turnTimer = time.AfterFunc(duration, func() {
		defer func() {
			if r := recover(); r != nil {
				log.Error().
					Interface("panic", r).
					Str("game_id", g.id).
					Msg("Turn timer goroutine panicked")
			}
		}()

		g.processTurnTimeout(ctx, duration, server)
	})
}

// processTurnTimeout handles turn timeout logic
func (g *gameInstance) processTurnTimeout(ctx context.Context, duration time.Duration, server *Server) {
	// Turn timeout - submit nil actions for players who haven't acted
	g.mu.Lock()

	// Fill in nil actions for missing players
	for _, player := range g.players {
		if _, exists := g.actionBuffer[player.id]; !exists {
			g.actionBuffer[player.id] = nil
		}
	}

	allSubmitted := len(g.actionBuffer) >= len(g.players)
	g.mu.Unlock()

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

// GetActiveGoroutineCount returns an estimate of active goroutines for this game manager
func (gm *GameManager) GetActiveGoroutineCount() int {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	// Count: cleanup goroutine + aggregator (if active) + games with active timers
	count := 1 // cleanup goroutine

	if gm.experienceAggregator != nil {
		count++ // aggregator goroutine
	}

	// Each active game may have timer goroutines and experience collection goroutines
	for _, game := range gm.games {
		if game.CurrentPhase() == commonv1.GamePhase_GAME_PHASE_RUNNING {
			count += 2 // timer + experience collection
		}
	}

	return count
}
