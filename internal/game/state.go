package game

import (
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

type Player struct {
	ID         int
	Alive      bool
	ArmyCount  int // cached every turn
	GeneralIdx int // -1 if eliminated
	OwnedTiles []int // indices of tiles owned by this player
}

type GameState struct {
	Turn            int
	Board           *core.Board
	Players         []Player
	ChangedTiles    map[int]struct{}
	FogOfWarEnabled bool
	// VisibilityChanges tracks tiles whose ownership changed, affecting visibility
	// Used for incremental fog of war updates
	VisibilityChangedTiles map[int]struct{}
}