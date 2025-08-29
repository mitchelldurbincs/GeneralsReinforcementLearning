package game

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/mapgen"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/processor"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/rules"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/states"
	"github.com/rs/zerolog"
)

// EngineInitializer handles the complex initialization of a game engine
type EngineInitializer struct {
	config GameConfig
	logger zerolog.Logger
}

// NewEngineInitializer creates a new engine initializer
func NewEngineInitializer(cfg GameConfig) *EngineInitializer {
	logger := cfg.Logger.With().Str("component", "GameEngine").Logger()
	return &EngineInitializer{
		config: cfg,
		logger: logger,
	}
}

// Initialize creates and initializes a new game engine
func (ei *EngineInitializer) Initialize(ctx context.Context) (*Engine, error) {
	// Check context early
	select {
	case <-ctx.Done():
		ei.logger.Error().Err(ctx.Err()).Msg("Engine creation cancelled or timed out during initial phase")
		return nil, ctx.Err()
	default:
	}

	// Setup configuration defaults
	ei.setupDefaults()

	// Generate the game map
	board, err := ei.generateMap()
	if err != nil {
		return nil, fmt.Errorf("map generation failed: %w", err)
	}

	// Initialize game state
	gs := ei.initializeGameState(board)

	// Initialize players
	ei.initializePlayers(gs, board)

	// Create engine components
	engine := ei.createEngine(gs)

	// Setup event handling
	ei.setupEventHandling(engine)

	// Perform initial game setup
	ei.performInitialSetup(engine)

	// Initialize state machine
	if err := ei.initializeStateMachine(engine); err != nil {
		return nil, fmt.Errorf("state machine initialization failed: %w", err)
	}

	// Publish game started event
	engine.eventBus.Publish(events.NewGameStartedEvent(
		engine.gameID,
		ei.config.Players,
		ei.config.Width,
		ei.config.Height,
	))

	ei.logger.Info().
		Int("width", ei.config.Width).
		Int("height", ei.config.Height).
		Int("players", ei.config.Players).
		Msg("Engine created successfully")

	return engine, nil
}

// setupDefaults sets up default values for missing configuration
func (ei *EngineInitializer) setupDefaults() {
	if ei.config.Rng == nil {
		ei.logger.Debug().Msg("No RNG provided, creating new seeded RNG")
		ei.config.Rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	if ei.config.GameID == "" {
		ei.config.GameID = fmt.Sprintf("game_%d", time.Now().UnixNano())
	}

	if ei.config.ExperienceCollector != nil {
		ei.logger.Info().Msg("Experience collection enabled")
	}
}

// generateMap generates the game map
func (ei *EngineInitializer) generateMap() (*core.Board, error) {
	mapCfg := mapgen.DefaultMapConfig(ei.config.Width, ei.config.Height, ei.config.Players)
	generator := mapgen.NewGenerator(mapCfg, ei.config.Rng)
	return generator.GenerateMap()
}

// initializeGameState creates the initial game state
func (ei *EngineInitializer) initializeGameState(board *core.Board) *GameState {
	return &GameState{
		Board:                  board,
		Players:                make([]Player, ei.config.Players),
		Turn:                   0,
		FogOfWarEnabled:        true,
		ChangedTiles:           make(map[int]struct{}),
		VisibilityChangedTiles: make(map[int]struct{}),
	}
}

// initializePlayers sets up all players and finds their generals
func (ei *EngineInitializer) initializePlayers(gs *GameState, board *core.Board) {
	for i := 0; i < ei.config.Players; i++ {
		generalIdx := ei.findPlayerGeneral(board, i)
		
		// Get the actual army count from the general tile
		armyCount := 1  // Default if general not found
		if generalIdx >= 0 && generalIdx < len(board.T) {
			armyCount = board.T[generalIdx].Army
		}

		gs.Players[i] = Player{
			ID:         i,
			Alive:      true,
			GeneralIdx: generalIdx,
			ArmyCount:  armyCount,
			OwnedTiles: make([]int, 0, 50), // Pre-allocate some capacity
		}
	}
}

// findPlayerGeneral finds the general tile for a specific player
func (ei *EngineInitializer) findPlayerGeneral(board *core.Board, playerID int) int {
	for idx, tile := range board.T {
		if tile.Type == core.TileGeneral && tile.Owner == playerID {
			return idx
		}
	}
	return -1
}

// createEngine creates the engine with all its components
func (ei *EngineInitializer) createEngine(gs *GameState) *Engine {
	// Create action processor
	actionProc := processor.NewActionProcessor(ei.logger)

	// Create event bus
	eventBus := events.NewEventBus()

	// Create game context for state machine
	gameContext := states.NewGameContext(ei.config.GameID, ei.config.Players, ei.logger)
	gameContext.PlayerCount = ei.config.Players

	// Create state machine
	stateMachine := states.NewStateMachine(gameContext, eventBus)

	engine := &Engine{
		gs:                  gs,
		rng:                 ei.config.Rng,
		gameOver:            false,
		logger:              ei.logger,
		actionProcessor:     actionProc,
		winCondition:        rules.NewWinConditionChecker(ei.logger, ei.config.Players),
		legalMoves:          rules.NewLegalMoveCalculator(),
		eventBus:            eventBus,
		gameID:              ei.config.GameID,
		stateMachine:        stateMachine,
		experienceCollector: ei.config.ExperienceCollector,
		tempTileOwnership:   make(map[int]int),
		tempAffectedPlayers: make(map[int]struct{}),
	}

	// Create managers after engine is created
	engine.productionManager = NewProductionManager(eventBus, ei.config.GameID, ei.logger)
	engine.turnProcessor = NewTurnProcessor(engine)

	return engine
}

// setupEventHandling configures event handling for the engine
func (ei *EngineInitializer) setupEventHandling(engine *Engine) {
	// Set the event publisher on the action processor using an adapter
	eventAdapter := events.NewEventPublisherAdapter(engine.eventBus)
	engine.actionProcessor.SetEventPublisher(eventAdapter)
}

// performInitialSetup performs initial game setup
func (ei *EngineInitializer) performInitialSetup(engine *Engine) {
	// Initial update of player stats to populate OwnedTiles
	engine.updatePlayerStats()
	engine.updateFogOfWar()

	// Check game over (probably not needed unless a 0-player game is valid)
	engine.checkGameOver(ei.logger.With().Str("phase", "init").Logger())
}

// initializeStateMachine sets up the state machine transitions
func (ei *EngineInitializer) initializeStateMachine(engine *Engine) error {
	stateMachine := engine.stateMachine
	gameContext := stateMachine.GetContext()

	// Transition through initial states
	// First move to Lobby state
	if err := stateMachine.TransitionTo(states.PhaseLobby, "Engine initialized"); err != nil {
		ei.logger.Error().Err(err).Msg("Failed to transition to Lobby state")
		return err
	}

	// Since all players are already added during map generation, transition to Starting
	if err := stateMachine.TransitionTo(states.PhaseStarting, "All players ready"); err != nil {
		ei.logger.Error().Err(err).Msg("Failed to transition to Starting state")
		return err
	}

	// Map is generated and players are placed, transition to Running
	gameContext.StartTime = time.Now() // Set start time when transitioning to running
	if err := stateMachine.TransitionTo(states.PhaseRunning, "Game setup complete"); err != nil {
		ei.logger.Error().Err(err).Msg("Failed to transition to Running state")
		return err
	}

	return nil
}
