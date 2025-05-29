package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

func main() {
	// Uncomment one of these to choose your testing mode:
	
	// Option 1: Random action simulation
	randomActionDemo()
	
	// Option 2: Manual testing with fixed moves
	// manualTest()
}

func randomActionDemo() {
	// Create a seeded RNG for reproducible games during development
	seed := time.Now().UnixNano()
	fmt.Printf("Game seed: %d\n", seed)
	rng := rand.New(rand.NewSource(seed))
	
	// Create a 8x8 game with 2 players for more interesting gameplay
	g := game.NewEngine(8, 8, 2, rng)
	
	fmt.Printf("Initial board:\n%s\n", g.Board())
	fmt.Printf("Player stats: %+v\n\n", g.GameState().Players)
	
	// Run game simulation
	maxTurns := 50
	for turn := 0; turn < maxTurns && !g.IsGameOver(); turn++ {
		// Generate some random actions for demonstration
		actions := generateRandomActions(g, rng)
		
		// Apply actions and advance game
		if err := g.Step(actions); err != nil {
			fmt.Printf("Error on turn %d: %v\n", turn+1, err)
			break
		}
		
		// Print results every few turns
		if turn%5 == 0 || len(actions) > 0 {
			fmt.Printf("Turn %d (actions: %d):\n", turn+1, len(actions))
			if len(actions) > 0 {
				for _, action := range actions {
					if moveAction, ok := action.(*core.MoveAction); ok {
						fmt.Printf("  Player %d: (%d,%d) -> (%d,%d) [%s]\n", 
							moveAction.PlayerID, 
							moveAction.FromX, moveAction.FromY,
							moveAction.ToX, moveAction.ToY,
							map[bool]string{true: "all", false: "half"}[moveAction.MoveAll])
					}
				}
			}
			fmt.Printf("%s", g.Board())
			
			state := g.GameState()
			for _, player := range state.Players {
				status := "ALIVE"
				if !player.Alive {
					status = "DEAD"
				}
				fmt.Printf("Player %d: %d armies, %s\n", player.ID, player.ArmyCount, status)
			}
			fmt.Println()
		}
	}
	
	// Game over
	if g.IsGameOver() {
		winner := g.GetWinner()
		if winner >= 0 {
			fmt.Printf("🎉 Game Over! Player %d wins!\n", winner)
		} else {
			fmt.Printf("Game Over! No winner.\n")
		}
	} else {
		fmt.Printf("Game reached maximum turns (%d)\n", maxTurns)
	}
	
	fmt.Printf("\nFinal board:\n%s", g.Board())
}

// generateRandomActions creates some random actions for alive players
// This is just for demonstration - in a real game, actions would come from players/AI
func generateRandomActions(g *game.Engine, rng *rand.Rand) []core.Action {
	var actions []core.Action
	state := g.GameState()
	
	for _, player := range state.Players {
		if !player.Alive {
			continue
		}
		
		// 30% chance this player takes an action this turn
		if rng.Float32() > 0.3 {
			continue
		}
		
		// Find tiles owned by this player that can move
		var validMoves []core.MoveAction
		board := state.Board
		
		for y := 0; y < board.H; y++ {
			for x := 0; x < board.W; x++ {
				idx := board.Idx(x, y)
				tile := board.T[idx]
				
				if tile.Owner != player.ID || tile.Army <= 1 {
					continue
				}
				
				// Check all 4 adjacent tiles
				directions := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
				for _, dir := range directions {
					toX, toY := x+dir[0], y+dir[1]
					
					// Check bounds
					if toX < 0 || toX >= board.W || toY < 0 || toY >= board.H {
						continue
					}
					
					move := core.MoveAction{
						PlayerID: player.ID,
						FromX:    x,
						FromY:    y,
						ToX:      toX,
						ToY:      toY,
						MoveAll:  rng.Float32() < 0.7, // 70% chance to move all
					}
					
					// Validate the move
					if move.Validate(board, player.ID) == nil {
						validMoves = append(validMoves, move)
					}
				}
			}
		}
		
		// Pick a random valid move
		if len(validMoves) > 0 {
			chosen := validMoves[rng.Intn(len(validMoves))]
			actions = append(actions, &chosen)
		}
	}
	
	return actions
}