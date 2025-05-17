package main

import (
    "fmt"
    "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
)

func main() {
    // Quick demo: create a 5×5 board with 2 players and run 10 ticks
    g := game.NewEngine(5, 5, 2)
    for turn := 0; turn < 10; turn++ {
        g.Step()
        fmt.Printf("After turn %d:\n%s\n", turn+1, g.Board())
    }
}