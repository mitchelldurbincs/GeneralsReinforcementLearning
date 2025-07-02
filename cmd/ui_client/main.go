package main

import (
	"context"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/ui"
	"github.com/rs/zerolog"
)

func main() {
	// Initialize your game engine (this is just an example setup)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()
	gameEngine := game.NewEngine(context.Background(), 20, 15, 2, rng, logger) // Example: 20x15 map, 2 players

	uiGame, err := ui.NewUIGame(gameEngine)
	if err != nil {
		log.Fatal(err)
	}

	ebiten.SetWindowSize(ui.ScreenWidth, ui.ScreenHeight)
	ebiten.SetWindowTitle("Generals RL UI")

	if err := ebiten.RunGame(uiGame); err != nil {
		log.Fatal(err)
	}
}
