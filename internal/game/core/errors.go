package core

import (
	"errors"
	"fmt"
)

var (
	ErrInvalidCoordinates = errors.New("invalid coordinates")
	ErrNotAdjacent        = errors.New("tiles are not adjacent")
	ErrNotOwned           = errors.New("tile not owned by player")
	ErrInsufficientArmy   = errors.New("insufficient army to move")
	ErrGameOver           = errors.New("game is over")
	ErrInvalidPlayer      = errors.New("invalid player ID")
	ErrMoveToSelf         = errors.New("cannot move to the same tile")
	ErrTargetIsMountain   = errors.New("target tile is a mountain")
)

// WrapActionError wraps an error with action context information
func WrapActionError(action Action, err error) error {
	if err == nil {
		return nil
	}

	switch a := action.(type) {
	case *MoveAction:
		return fmt.Errorf("player %d: move from (%d,%d) to (%d,%d): %w",
			a.PlayerID, a.FromX, a.FromY, a.ToX, a.ToY, err)
	default:
		return fmt.Errorf("player action: %w", err)
	}
}

// WrapGameStateError wraps an error with game state context
func WrapGameStateError(turn int, phase string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("game turn %d [%s]: %w", turn, phase, err)
}

// WrapPlayerError wraps an error with player context
func WrapPlayerError(playerID int, operation string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("player %d %s: %w", playerID, operation, err)
}

// GameError represents a structured error with game context
type GameError struct {
	Turn      int
	PlayerID  int
	Operation string
	Err       error
}

// Error implements the error interface
func (e *GameError) Error() string {
	if e.PlayerID != 0 {
		return fmt.Sprintf("turn %d: player %d %s: %v", e.Turn, e.PlayerID, e.Operation, e.Err)
	}
	return fmt.Sprintf("turn %d: %s: %v", e.Turn, e.Operation, e.Err)
}

// Unwrap allows errors.Is and errors.As to work with GameError
func (e *GameError) Unwrap() error {
	return e.Err
}

// NewGameError creates a new GameError
func NewGameError(turn, playerID int, operation string, err error) *GameError {
	return &GameError{
		Turn:      turn,
		PlayerID:  playerID,
		Operation: operation,
		Err:       err,
	}
}
