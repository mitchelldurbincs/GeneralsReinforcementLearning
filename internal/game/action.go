package game

import "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"

// ActionType represents the type of action
type ActionType int

const (
	ActionTypeMove ActionType = iota
	ActionTypeNoOp
)

// Action represents a player action in the game
type Action struct {
	Type ActionType
	From core.Coordinate
	To   core.Coordinate
}
