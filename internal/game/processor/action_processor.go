package processor

import (
	"context"
	"sort"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/rs/zerolog"
)

// ActionProcessor handles the processing of player actions during each game tick
type ActionProcessor struct {
	logger zerolog.Logger
}

// NewActionProcessor creates a new action processor
func NewActionProcessor(logger zerolog.Logger) *ActionProcessor {
	return &ActionProcessor{
		logger: logger.With().Str("component", "ActionProcessor").Logger(),
	}
}

// ProcessActions processes all actions for the current tick
// Returns capture details, affected visibility tiles, and any error encountered
func (ap *ActionProcessor) ProcessActions(ctx context.Context, board *core.Board, players []PlayerInfo, actions []core.Action, changedTiles map[int]struct{}) ([]core.CaptureDetails, map[int]struct{}, error) {
	ap.logger.Debug().Msg("Sorting actions for deterministic processing")
	// Sort actions by player ID for deterministic processing
	sort.Slice(actions, func(i, j int) bool {
		return actions[i].GetPlayerID() < actions[j].GetPlayerID()
	})

	var encounteredError error
	var allCaptureDetailsThisTurn []core.CaptureDetails
	visibilityChangedTiles := make(map[int]struct{})

	for _, action := range actions {
		// Check context before processing each action
		select {
		case <-ctx.Done():
			ap.logger.Warn().Err(ctx.Err()).Msg("Action processing interrupted by context cancellation")
			return allCaptureDetailsThisTurn, visibilityChangedTiles, ctx.Err()
		default:
		}

		playerID := action.GetPlayerID()
		if playerID < 0 || playerID >= len(players) || !players[playerID].IsAlive() {
			ap.logger.Warn().Int("player_id", playerID).Bool("alive", playerID >= 0 && playerID < len(players) && players[playerID].IsAlive()).Msg("Ignoring action from invalid or dead player")
			continue
		}

		ap.logger.Debug().Int("player_id", playerID).Interface("action", action).Msg("Applying action")
		switch act := action.(type) {
		case *core.MoveAction:
			captureDetail, err := core.ApplyMoveAction(board, act, changedTiles)
			if err != nil {
				// Wrap the error with action context for better debugging
				wrappedErr := core.WrapActionError(act, err)
				ap.logger.Error().Err(wrappedErr).
					Int("player_id", playerID).
					Interface("action_details", act).
					Msg("Failed to apply move action")
				if encounteredError == nil {
					encounteredError = wrappedErr
				}
				continue
			}
			if captureDetail != nil {
				ap.logger.Debug().
					Int("player_id", playerID).
					Interface("capture_details", *captureDetail).
					Msg("Move resulted in capture")
				allCaptureDetailsThisTurn = append(allCaptureDetailsThisTurn, *captureDetail)
				// Mark tile for visibility update since ownership changed
				capturedTileIdx := board.Idx(captureDetail.X, captureDetail.Y)
				visibilityChangedTiles[capturedTileIdx] = struct{}{}
			}
		default:
			ap.logger.Warn().Int("player_id", playerID).Str("action_type", core.GetActionType(action)).Msg("Unhandled action type in processActions")
		}
	}

	// If there was an error, wrap it with additional context
	if encounteredError != nil {
		// Note: encounteredError already has action context from WrapActionError above
		return allCaptureDetailsThisTurn, visibilityChangedTiles, encounteredError
	}
	return allCaptureDetailsThisTurn, visibilityChangedTiles, nil
}

// PlayerInfo interface - matches the Player struct from game package
// This avoids circular imports
type PlayerInfo interface {
	GetID() int
	IsAlive() bool
}