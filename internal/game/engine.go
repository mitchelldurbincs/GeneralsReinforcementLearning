package game

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/processor"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/rules"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/states"
	"github.com/rs/zerolog"
)

type Engine struct {
	gs       *GameState
	rng      *rand.Rand
	gameOver bool
	logger   zerolog.Logger
	
	// Extracted components
	actionProcessor    *processor.ActionProcessor
	winCondition       *rules.WinConditionChecker
	legalMoves         *rules.LegalMoveCalculator
	turnProcessor      *TurnProcessor      // Added turn processor
	productionManager  *ProductionManager  // Added production manager
	
	// Event system
	eventBus *events.EventBus
	gameID   string
	
	// State management
	stateMachine *states.StateMachine
	
	// Experience collection
	experienceCollector ExperienceCollector
	
	// Reusable temporary maps to avoid allocations
	tempTileOwnership   map[int]int        // Used in performIncrementalStatsUpdate
	tempAffectedPlayers map[int]struct{}   // Used in performIncrementalVisibilityUpdate
}

type GameConfig struct {
	Width               int
	Height              int
	Players             int
	Rng                 *rand.Rand
	Logger              zerolog.Logger
	GameID              string // Optional game ID, will be generated if not provided
	ExperienceCollector ExperienceCollector // Optional experience collector
}

// NewEngine creates a new game engine with map generation.
// Accepts a context, primarily for consistency and potential future use (e.g., complex setup).
func NewEngine(ctx context.Context, cfg GameConfig) *Engine {
	initializer := NewEngineInitializer(cfg)
	engine, err := initializer.Initialize(ctx)
	if err != nil {
		// Log error and return nil to maintain backward compatibility
		cfg.Logger.Error().Err(err).Msg("Failed to initialize engine")
		return nil
	}
	return engine
}

// Step processes actions and advances the game by one turn.
// It accepts a context for cancellation/timeout.
func (e *Engine) Step(ctx context.Context, actions []core.Action) error {
	return e.turnProcessor.ProcessTurn(ctx, actions)
}

// processActions handles all actions for this turn using the ActionProcessor.
func (e *Engine) processActions(ctx context.Context, actions []core.Action, l zerolog.Logger) error {
	// Publish ActionSubmitted events for each action
	for _, action := range actions {
		e.eventBus.Publish(events.NewActionSubmittedEvent(e.gameID, action.GetPlayerID(), action, e.gs.Turn))
	}

	// Create a slice of PlayerInfo interfaces from our Players
	playerInfos := make([]processor.PlayerInfo, len(e.gs.Players))
	for i := range e.gs.Players {
		playerInfos[i] = &e.gs.Players[i]
	}
	
	// Use the ActionProcessor to handle all action processing
	captureDetails, visibilityChangedTiles, err := e.actionProcessor.ProcessActions(ctx, e.gs.Board, playerInfos, actions, e.gs.ChangedTiles)
	
	// Merge visibility changed tiles
	for tileIdx := range visibilityChangedTiles {
		e.gs.VisibilityChangedTiles[tileIdx] = struct{}{}
	}
	
	// Handle eliminations if there were captures
	if len(captureDetails) > 0 {
		l.Debug().Int("num_captures_this_turn", len(captureDetails)).Msg("Processing captures for eliminations")
		eliminationOrders := core.ProcessCaptures(captureDetails)
		if len(eliminationOrders) > 0 {
			e.handleEliminationsAndTileTurnover(eliminationOrders, l)
			// Update player stats immediately after eliminations so production is applied correctly
			e.updatePlayerStats()
		}
	}

	if err != nil {
		return core.WrapGameStateError(e.gs.Turn, "processing actions", err)
	}
	return nil
}

// handleEliminationsAndTileTurnover processes player eliminations and transfers tiles.
func (e *Engine) handleEliminationsAndTileTurnover(orders []core.PlayerEliminationOrder, l zerolog.Logger) {
	l.Info().Int("num_elimination_orders", len(orders)).Msg("Handling player eliminations and tile turnover")
	for _, order := range orders {
		l.Info().
			Int("eliminated_player_id", order.EliminatedPlayerID).
			Int("new_owner_player_id", order.NewOwnerID).
			Msg("Player eliminated, transferring assets")

		// Use the owned tiles list for efficient transfer
		eliminatedPlayer := &e.gs.Players[order.EliminatedPlayerID]
		tilesTransferred := 0
		
		for _, tileIdx := range eliminatedPlayer.OwnedTiles {
			if e.gs.Board.T[tileIdx].Owner == order.EliminatedPlayerID {
				e.gs.Board.T[tileIdx].Owner = order.NewOwnerID
				e.gs.ChangedTiles[tileIdx] = struct{}{}
				e.gs.VisibilityChangedTiles[tileIdx] = struct{}{}
				tilesTransferred++
			}
		}
		
		// Mark the player as eliminated
		eliminatedPlayer.Alive = false
		eliminatedPlayer.GeneralIdx = -1
		
		// Publish PlayerEliminated event
		e.eventBus.Publish(events.NewPlayerEliminatedEvent(e.gameID, order.EliminatedPlayerID, order.NewOwnerID, 0, e.gs.Turn))
		
		l.Debug().
			Int("eliminated_player_id", order.EliminatedPlayerID).
			Int("new_owner_player_id", order.NewOwnerID).
			Int("tiles_transferred", tilesTransferred).
			Msg("Asset transfer details")
	}
}

// processTurnProduction applies army growth
func (e *Engine) processTurnProduction(l zerolog.Logger) {
	e.productionManager.ProcessTurnProduction(e.gs, e.gs.Turn)
}

// checkGameOver determines if the game is over using the WinConditionChecker.
func (e *Engine) checkGameOver(l zerolog.Logger) {
	// Create a slice of Player interfaces for the win condition checker
	players := make([]rules.Player, len(e.gs.Players))
	for i := range e.gs.Players {
		players[i] = &e.gs.Players[i]
	}
	
	wasGameOver := e.gameOver
	gameOver, winnerID := e.winCondition.CheckGameOver(players)
	e.gameOver = gameOver
	
	if !wasGameOver && e.gameOver {
		l.Info().Msg("Game over condition met")
		
		// Update game context with winner
		e.stateMachine.GetContext().Winner = winnerID
		
		// Transition to Ending state
		if err := e.stateMachine.TransitionTo(states.PhaseEnding, "Game over condition met"); err != nil {
			l.Error().Err(err).Msg("Failed to transition to Ending state")
		}
		
		// Transition to Ended state
		if err := e.stateMachine.TransitionTo(states.PhaseEnded, "Game finalized"); err != nil {
			l.Error().Err(err).Msg("Failed to transition to Ended state")
		}
		
		// Publish GameEnded event
		// TODO: Track game start time to calculate duration
		e.eventBus.Publish(events.NewGameEndedEvent(e.gameID, winnerID, 0, e.gs.Turn))
	} else if wasGameOver && !e.gameOver {
		// This would be highly unusual
		l.Error().Msg("Game was over, but is no longer over. This is unexpected!")
	}
}

// Public accessors
func (e *Engine) GameState() GameState { return *e.gs }
func (e *Engine) IsGameOver() bool   { return e.gameOver }
func (e *Engine) EventBus() *events.EventBus { return e.eventBus }
func (e *Engine) CurrentPhase() states.GamePhase { return e.stateMachine.CurrentPhase() }

// Pause pauses the game if it's currently running
func (e *Engine) Pause(reason string) error {
	currentPhase := e.stateMachine.CurrentPhase()
	if currentPhase != states.PhaseRunning {
		return fmt.Errorf("cannot pause game in %s phase", currentPhase)
	}
	
	// Update pause time in context
	e.stateMachine.GetContext().PauseTime = time.Now()
	
	// Transition to paused state
	if err := e.stateMachine.TransitionTo(states.PhasePaused, reason); err != nil {
		e.logger.Error().Err(err).Msg("Failed to pause game")
		return err
	}
	
	e.logger.Info().Str("reason", reason).Msg("Game paused")
	return nil
}

// Resume resumes the game if it's currently paused
func (e *Engine) Resume(reason string) error {
	currentPhase := e.stateMachine.CurrentPhase()
	if currentPhase != states.PhasePaused {
		return fmt.Errorf("cannot resume game in %s phase", currentPhase)
	}
	
	// Update total pause duration
	ctx := e.stateMachine.GetContext()
	if !ctx.PauseTime.IsZero() {
		pauseDuration := time.Since(ctx.PauseTime)
		ctx.TotalPauseDuration += pauseDuration
		ctx.PauseTime = time.Time{} // Reset pause time
	}
	
	// Transition back to running state
	if err := e.stateMachine.TransitionTo(states.PhaseRunning, reason); err != nil {
		e.logger.Error().Err(err).Msg("Failed to resume game")
		return err
	}
	
	e.logger.Info().Str("reason", reason).Msg("Game resumed")
	return nil
}

// GetWinner returns the winning player ID, or -1 if game isn't over or it's a draw
func (e *Engine) GetWinner() int {
	e.logger.Debug().Bool("is_game_over_state", e.gameOver).Msg("GetWinner called")
	if !e.gameOver {
		e.logger.Warn().Msg("GetWinner called but game is not actually over according to internal state.")
		return -1
	}
	
	// Create a slice of Player interfaces for the win condition checker
	players := make([]rules.Player, len(e.gs.Players))
	for i := range e.gs.Players {
		players[i] = &e.gs.Players[i]
	}
	
	_, winnerID := e.winCondition.CheckGameOver(players)
	return winnerID
}

// GetLegalActionMask returns a flattened boolean mask indicating which actions are legal for the given player.
// For a board of width W and height H:
// - Total actions = W * H * 4 (4 directions per tile)  
// - Index = (y * W + x) * 4 + direction
// - Directions: 0=up, 1=right, 2=down, 3=left
// - true = legal move, false = illegal move
func (e *Engine) GetLegalActionMask(playerID int) []bool {
	// If player is not alive or invalid, return empty mask
	if playerID < 0 || playerID >= len(e.gs.Players) {
		maskSize := e.gs.Board.W * e.gs.Board.H * 4
		return make([]bool, maskSize)
	}
	
	player := &e.gs.Players[playerID]
	return e.legalMoves.GetLegalActionMask(e.gs.Board, player, player.OwnedTiles)
}

// GetChangedTiles returns the set of tiles that changed in the last turn
func (e *Engine) GetChangedTiles() map[int]bool {
	result := make(map[int]bool, len(e.gs.ChangedTiles))
	for tileIdx := range e.gs.ChangedTiles {
		result[tileIdx] = true
	}
	return result
}

// GetVisibilityChangedTiles returns the set of tiles whose visibility changed in the last turn
func (e *Engine) GetVisibilityChangedTiles() map[int]bool {
	result := make(map[int]bool, len(e.gs.VisibilityChangedTiles))
	for tileIdx := range e.gs.VisibilityChangedTiles {
		result[tileIdx] = true
	}
	return result
}

// GetExperienceCollector returns the experience collector if available
func (e *Engine) GetExperienceCollector() ExperienceCollector {
	return e.experienceCollector
}