package testutil

import (
	"github.com/your-org/generals/internal/game/core"
)

// CreateTestBoard creates a test board with the given dimensions
func CreateTestBoard(width, height int) *core.Board {
	board := core.NewBoard(width, height)
	return board
}

// CreateTestBoardWithTiles creates a test board and sets up specific tiles
func CreateTestBoardWithTiles(width, height int, tiles map[core.Coordinate]core.Tile) *core.Board {
	board := core.NewBoard(width, height)
	for coord, tile := range tiles {
		board.SetTile(coord, tile)
	}
	return board
}

// CreateTestPlayers creates a slice of test players
func CreateTestPlayers(count int) []*core.Player {
	players := make([]*core.Player, count)
	colors := []string{"red", "blue", "green", "yellow"}
	for i := 0; i < count; i++ {
		players[i] = &core.Player{
			ID:        i,
			Color:     colors[i%len(colors)],
			IsAlive:   true,
			TotalArmy: 1,
			TotalLand: 1,
		}
	}
	return players
}

// CreateSimpleTestSetup creates a simple 5x5 board with 2 players
// Player 0 (red) general at (1,1), Player 1 (blue) general at (3,3)
func CreateSimpleTestSetup() (*core.Board, []*core.Player) {
	board := CreateTestBoard(5, 5)
	players := CreateTestPlayers(2)
	
	// Place generals
	board.SetTile(core.Coordinate{X: 1, Y: 1}, core.Tile{
		Type:    core.General,
		OwnerID: 0,
		Army:    1,
	})
	board.SetTile(core.Coordinate{X: 3, Y: 3}, core.Tile{
		Type:    core.General,
		OwnerID: 1,
		Army:    1,
	})
	
	return board, players
}