package gameserver

import (
	"context"
	"fmt"

	"github.com/rs/zerolog/log"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// ValidationResult contains the outcome of a validation check
type ValidationResult struct {
	Valid          bool
	ErrorCode      commonv1.ErrorCode
	ErrorMessage   string
	CachedResponse *gamev1.SubmitActionResponse
}

// ActionValidator handles all action validation logic
type ActionValidator struct {
	gameManager *GameManager
}

// NewActionValidator creates a new validator instance
func NewActionValidator(gm *GameManager) *ActionValidator {
	return &ActionValidator{
		gameManager: gm,
	}
}

// ValidateSubmitActionRequest performs complete validation for action submission
func (v *ActionValidator) ValidateSubmitActionRequest(
	ctx context.Context,
	req *gamev1.SubmitActionRequest,
) (*ValidationResult, *gameInstance) {

	// 1. Validate game exists
	game, exists := v.gameManager.GetGame(req.GameId)
	if !exists {
		return &ValidationResult{
			Valid:        false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_GAME_NOT_FOUND,
			ErrorMessage: fmt.Sprintf("game %s not found", req.GameId),
		}, nil
	}

	// 2. Check idempotency
	if req.IdempotencyKey != "" {
		if cached := game.idempotencyManager.Check(req.IdempotencyKey); cached != nil {
			log.Debug().
				Str("game_id", req.GameId).
				Str("idempotency_key", req.IdempotencyKey).
				Msg("Returning cached response for idempotent request")
			return &ValidationResult{
				Valid:          false,
				CachedResponse: cached,
			}, game
		}
	}

	// 3. Validate game phase
	currentPhase := game.CurrentPhase()
	if currentPhase != commonv1.GamePhase_GAME_PHASE_RUNNING {
		errorCode := commonv1.ErrorCode_ERROR_CODE_INVALID_PHASE
		if currentPhase == commonv1.GamePhase_GAME_PHASE_ENDED {
			errorCode = commonv1.ErrorCode_ERROR_CODE_GAME_OVER
		}

		return &ValidationResult{
			Valid:        false,
			ErrorCode:    errorCode,
			ErrorMessage: fmt.Sprintf("game %s cannot accept actions in %s phase", req.GameId, currentPhase.String()),
		}, game
	}

	// 4. Authenticate player
	authenticated := false
	for _, p := range game.players {
		if p.id == req.PlayerId && p.token == req.PlayerToken {
			authenticated = true
			break
		}
	}

	if !authenticated {
		return &ValidationResult{
			Valid:        false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_PLAYER,
			ErrorMessage: fmt.Sprintf("invalid player credentials for game %s: player %d", req.GameId, req.PlayerId),
		}, game
	}

	// 5. Validate turn number
	game.mu.RLock()
	currentTurn := game.currentTurn
	game.mu.RUnlock()

	if req.Action != nil && req.Action.TurnNumber != currentTurn {
		return &ValidationResult{
			Valid:        false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
			ErrorMessage: fmt.Sprintf("invalid turn number for game %s: expected %d, got %d", req.GameId, currentTurn, req.Action.TurnNumber),
		}, game
	}

	return &ValidationResult{Valid: true}, game
}

// ValidateCoreAction validates the actual game action
func (v *ActionValidator) ValidateCoreAction(
	action core.Action,
	game *gameInstance,
	playerID int32,
	currentTurn int32,
) *ValidationResult {

	if action == nil {
		return &ValidationResult{Valid: true} // No action is valid (wait/skip)
	}

	game.mu.Lock()
	engineState := game.engine.GameState()
	err := action.Validate(engineState.Board, int(playerID))
	game.mu.Unlock()

	if err != nil {
		return &ValidationResult{
			Valid:        false,
			ErrorCode:    commonv1.ErrorCode_ERROR_CODE_INVALID_TURN,
			ErrorMessage: fmt.Sprintf("action validation failed for game %s player %d turn %d: %v", game.id, playerID, currentTurn, err),
		}
	}

	return &ValidationResult{Valid: true}
}
