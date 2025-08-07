package testutil

import (
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

// CreateTestBoard creates a test board with the given dimensions
func CreateTestBoard(width, height int) *core.Board {
	board := core.NewBoard(width, height)
	return board
}

// TilePosition represents x,y coordinates for a tile
type TilePosition struct {
	X, Y int
}

// CreateTestBoardWithTiles creates a test board and sets up specific tiles
func CreateTestBoardWithTiles(width, height int, tiles map[TilePosition]core.Tile) *core.Board {
	board := core.NewBoard(width, height)
	for pos, tile := range tiles {
		idx := board.Idx(pos.X, pos.Y)
		board.T[idx] = tile
	}
	return board
}

// CreateTestPlayers creates a slice of test players
func CreateTestPlayers(count int) []*game.Player {
	players := make([]*game.Player, count)
	for i := 0; i < count; i++ {
		players[i] = &game.Player{
			ID:         i,
			Alive:      true,
			ArmyCount:  1,
			GeneralIdx: -1, // Will be set when placing general
			OwnedTiles: []int{},
		}
	}
	return players
}

// CreateSimpleTestSetup creates a simple 5x5 board with 2 players
// Player 0 general at (1,1), Player 1 general at (3,3)
func CreateSimpleTestSetup() (*core.Board, []*game.Player) {
	board := CreateTestBoard(5, 5)
	players := CreateTestPlayers(2)

	// Place generals
	idx1 := board.Idx(1, 1)
	board.T[idx1] = core.Tile{
		Type:  core.TileGeneral,
		Owner: 0,
		Army:  1,
	}
	players[0].GeneralIdx = idx1
	players[0].OwnedTiles = append(players[0].OwnedTiles, idx1)

	idx2 := board.Idx(3, 3)
	board.T[idx2] = core.Tile{
		Type:  core.TileGeneral,
		Owner: 1,
		Army:  1,
	}
	players[1].GeneralIdx = idx2
	players[1].OwnedTiles = append(players[1].OwnedTiles, idx2)

	return board, players
}
