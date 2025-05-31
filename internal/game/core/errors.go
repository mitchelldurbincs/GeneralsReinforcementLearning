package core

import "errors"

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