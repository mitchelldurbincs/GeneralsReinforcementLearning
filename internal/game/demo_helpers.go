package game

import (
	"math/rand"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/rs/zerolog/log"
)

// GenerateRandomActions creates a set of random, valid moves for all active players.
// This is a helper function intended for demos, testing, or simple baseline agents.
func GenerateRandomActions(g *Engine, rng *rand.Rand) []core.Action {
	var actions []core.Action
	state := g.GameState()

	for _, player := range state.Players {
		if !player.Alive {
			continue
		}
		if rng.Float32() > 0.3 {
			continue
		}
		var validMoves []core.MoveAction
		board := state.Board
		for y := 0; y < board.H; y++ {
			for x := 0; x < board.W; x++ {
				idx := board.Idx(x, y)
				tile := board.T[idx]
				if tile.Owner != player.ID || tile.Army <= 1 {
					continue
				}
				directions := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
				for _, dir := range directions {
					toX, toY := x+dir[0], y+dir[1]
					if toX < 0 || toX >= board.W || toY < 0 || toY >= board.H {
						continue
					}
					targetIdx := board.Idx(toX, toY)
					if board.T[targetIdx].IsMountain() {
						continue
					}
					move := core.MoveAction{
						PlayerID: player.ID, FromX: x, FromY: y, ToX: toX, ToY: toY,
						MoveAll: rng.Float32() < 0.7,
					}
					validMoves = append(validMoves, move)
				}
			}
		}
		if len(validMoves) > 0 {
			chosen := validMoves[rng.Intn(len(validMoves))]
			actions = append(actions, &chosen)
			log.Debug().
					Int("player_id", player.ID).
					Int("from_x", chosen.FromX).Int("from_y", chosen.FromY).
					Int("to_x", chosen.ToX).Int("to_y", chosen.ToY).
					Bool("move_all", chosen.MoveAll).
					Msg("Generated random action")
		}
	}
	return actions
}
