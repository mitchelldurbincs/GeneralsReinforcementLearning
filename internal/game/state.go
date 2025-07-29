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

// GetID returns the player's ID
func (p *Player) GetID() int {
	return p.ID
}

// IsAlive returns whether the player is still alive
func (p *Player) IsAlive() bool {
	return p.Alive
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

// Clone creates a deep copy of the game state
func (gs *GameState) Clone() *GameState {
	clone := &GameState{
		Turn:            gs.Turn,
		Board:           gs.Board.Clone(),
		Players:         make([]Player, len(gs.Players)),
		ChangedTiles:    make(map[int]struct{}),
		FogOfWarEnabled: gs.FogOfWarEnabled,
		VisibilityChangedTiles: make(map[int]struct{}),
	}
	
	// Deep copy players
	for i, p := range gs.Players {
		clone.Players[i] = Player{
			ID:         p.ID,
			Alive:      p.Alive,
			ArmyCount:  p.ArmyCount,
			GeneralIdx: p.GeneralIdx,
			OwnedTiles: make([]int, len(p.OwnedTiles)),
		}
		copy(clone.Players[i].OwnedTiles, p.OwnedTiles)
	}
	
	// Copy changed tiles
	for k := range gs.ChangedTiles {
		clone.ChangedTiles[k] = struct{}{}
	}
	
	// Copy visibility changed tiles
	for k := range gs.VisibilityChangedTiles {
		clone.VisibilityChangedTiles[k] = struct{}{}
	}
	
	return clone
}

// IsGameOver returns true if the game has ended
func (gs *GameState) IsGameOver() bool {
	aliveCount := 0
	for _, p := range gs.Players {
		if p.Alive {
			aliveCount++
		}
	}
	return aliveCount <= 1
}

// GetWinner returns the ID of the winning player, or -1 if no winner yet
func (gs *GameState) GetWinner() int {
	var alivePlayer *Player
	aliveCount := 0
	
	for i := range gs.Players {
		if gs.Players[i].Alive {
			aliveCount++
			alivePlayer = &gs.Players[i]
		}
	}
	
	if aliveCount == 1 && alivePlayer != nil {
		return alivePlayer.ID
	}
	
	return -1
}