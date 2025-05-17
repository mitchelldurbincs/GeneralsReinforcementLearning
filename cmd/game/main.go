package main

import (
	"fmt"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"math/rand"
	"time"
)

func main() {
	// ① seed once
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// ② pass rng into NewEngine
	g := game.NewEngine(5, 5, 2, rng)

	for turn := 0; turn < 10; turn++ {
		g.Step()
		fmt.Printf("After turn %d:\n%s\n", turn+1, g.Board())
	}
}
