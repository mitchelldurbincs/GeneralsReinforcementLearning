package game

import (
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

type Player struct {
	ID         int
	Alive      bool
	ArmyCount  int // cached every turn
	GeneralIdx int // -1 if eliminated
}

type GameState struct {
	Turn    int
	Board   *core.Board
	Players []Player
}