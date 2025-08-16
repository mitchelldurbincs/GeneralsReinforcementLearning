package game

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/rs/zerolog"
)

// TurnProcessor handles the orchestration of a single turn
type TurnProcessor struct {
	engine *Engine
	logger zerolog.Logger
}

// NewTurnProcessor creates a new turn processor
func NewTurnProcessor(engine *Engine) *TurnProcessor {
	return &TurnProcessor{
		engine: engine,
		logger: engine.logger,
	}
}

// ProcessTurn executes a complete game turn
func (tp *TurnProcessor) ProcessTurn(ctx context.Context, actions []core.Action) error {
	// Check context at start
	if err := tp.checkContext(ctx, "before starting"); err != nil {
		return err
	}

	// Validate game state
	if err := tp.validateGameState(); err != nil {
		return err
	}

	// Capture state for experience collection
	prevState := tp.captureStateForExperience()

	// Initialize turn
	tp.initializeTurn()

	// Create turn-scoped logger
	turnLogger := tp.logger.With().Int("turn", tp.engine.gs.Turn).Logger()
	turnLogger.Debug().Msg("Starting game step")

	// Process the turn phases
	turnStartTime := time.Now()
	tp.publishTurnStarted()

	// Process actions phase
	if err := tp.processActionsPhase(ctx, actions, turnLogger); err != nil {
		return err
	}

	// Production phase
	if err := tp.processProductionPhase(ctx, turnLogger); err != nil {
		return err
	}

	// End of turn phase
	if err := tp.processEndOfTurnPhase(ctx, turnLogger); err != nil {
		return err
	}

	// Collect experiences if enabled
	tp.collectExperiences(prevState, actions)

	// Publish turn ended
	tp.publishTurnEnded(turnStartTime, len(actions))

	turnLogger.Debug().Msg("Game step finished")
	return nil
}

// checkContext checks if the context is cancelled
func (tp *TurnProcessor) checkContext(ctx context.Context, phase string) error {
	select {
	case <-ctx.Done():
		tp.logger.Warn().
			Err(ctx.Err()).
			Int("turn", tp.engine.gs.Turn).
			Str("phase", phase).
			Msg("Game step cancelled or timed out")
		return ctx.Err()
	default:
		return nil
	}
}

// validateGameState ensures the game can receive actions
func (tp *TurnProcessor) validateGameState() error {
	currentPhase := tp.engine.stateMachine.CurrentPhase()
	if !currentPhase.CanReceiveActions() {
		tp.logger.Warn().
			Str("current_phase", currentPhase.String()).
			Int("turn", tp.engine.gs.Turn).
			Msg("Attempted to step game in phase that cannot receive actions")
		return fmt.Errorf("game is in %s phase and cannot receive actions", currentPhase)
	}

	if tp.engine.gameOver {
		tp.logger.Warn().
			Int("turn", tp.engine.gs.Turn).
			Msg("Attempted to step game that is already over")
		return core.WrapGameStateError(tp.engine.gs.Turn, "step", core.ErrGameOver)
	}

	return nil
}

// captureStateForExperience captures the current state if experience collection is enabled
func (tp *TurnProcessor) captureStateForExperience() *GameState {
	if tp.engine.experienceCollector != nil {
		return tp.engine.gs.Clone()
	}
	return nil
}

// initializeTurn prepares for a new turn
func (tp *TurnProcessor) initializeTurn() {
	tp.engine.gs.Turn++
	tp.engine.updateFogOfWar()

	// Clear changed tiles from previous turn (reuse the map to avoid allocation)
	for k := range tp.engine.gs.ChangedTiles {
		delete(tp.engine.gs.ChangedTiles, k)
	}
	for k := range tp.engine.gs.VisibilityChangedTiles {
		delete(tp.engine.gs.VisibilityChangedTiles, k)
	}
}

// publishTurnStarted publishes the turn started event
func (tp *TurnProcessor) publishTurnStarted() {
	tp.engine.eventBus.Publish(events.NewTurnStartedEvent(tp.engine.gameID, tp.engine.gs.Turn))
}

// processActionsPhase handles action processing
func (tp *TurnProcessor) processActionsPhase(ctx context.Context, actions []core.Action, turnLogger zerolog.Logger) error {
	turnLogger.Debug().Int("num_actions_submitted", len(actions)).Msg("Processing actions")

	if err := tp.engine.processActions(ctx, actions, turnLogger); err != nil {
		// Check if the error is due to context cancellation
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return core.WrapGameStateError(tp.engine.gs.Turn, "action processing", fmt.Errorf("context cancelled: %w", err))
		}
		return core.WrapGameStateError(tp.engine.gs.Turn, "action processing", err)
	}

	turnLogger.Debug().Msg("Finished processing actions")
	return nil
}

// processProductionPhase handles production
func (tp *TurnProcessor) processProductionPhase(ctx context.Context, turnLogger zerolog.Logger) error {
	// Check context before potentially long operation
	if err := tp.checkContext(ctx, "before production"); err != nil {
		return core.WrapGameStateError(tp.engine.gs.Turn, "production phase", fmt.Errorf("context cancelled: %w", err))
	}

	tp.engine.processTurnProduction(turnLogger)
	return nil
}

// processEndOfTurnPhase handles end of turn updates
func (tp *TurnProcessor) processEndOfTurnPhase(ctx context.Context, turnLogger zerolog.Logger) error {
	// Check context before stats update
	if err := tp.checkContext(ctx, "before updating/checking stats"); err != nil {
		return core.WrapGameStateError(tp.engine.gs.Turn, "stats update", fmt.Errorf("context cancelled: %w", err))
	}

	tp.engine.updatePlayerStats()
	tp.engine.checkGameOver(turnLogger)
	return nil
}

// collectExperiences handles experience collection if enabled
func (tp *TurnProcessor) collectExperiences(prevState *GameState, actions []core.Action) {
	if tp.engine.experienceCollector == nil || prevState == nil {
		tp.logger.Debug().
			Bool("has_collector", tp.engine.experienceCollector != nil).
			Bool("has_prev_state", prevState != nil).
			Msg("Skipping experience collection")
		return
	}

	// Convert core.Action to game.Action with player IDs
	actionMap := make(map[int]*Action)
	for _, action := range actions {
		if moveAction, ok := action.(*core.MoveAction); ok {
			actionMap[moveAction.PlayerID] = &Action{
				Type: ActionTypeMove,
				From: moveAction.From,
				To:   moveAction.To,
			}
		}
	}

	tp.logger.Debug().
		Int("turn", tp.engine.gs.Turn).
		Int("num_actions", len(actionMap)).
		Msg("Collecting experiences for turn")

	tp.engine.experienceCollector.OnStateTransition(prevState, tp.engine.gs, actionMap)

	// If game is over, notify collector
	if tp.engine.gameOver {
		tp.logger.Info().
			Int("final_turn", tp.engine.gs.Turn).
			Msg("Game ended, notifying experience collector")
		tp.engine.experienceCollector.OnGameEnd(tp.engine.gs)
	}
}

// publishTurnEnded publishes the turn ended event
func (tp *TurnProcessor) publishTurnEnded(startTime time.Time, actionCount int) {
	tp.engine.eventBus.Publish(events.NewTurnEndedEvent(
		tp.engine.gameID,
		tp.engine.gs.Turn,
		actionCount,
		time.Since(startTime),
	))
}
