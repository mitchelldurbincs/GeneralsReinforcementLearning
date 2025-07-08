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
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()
	config := game.GameConfig{
		Width:   20,
		Height:  15,
		Players: 2,
		Rng:     rng,
		Logger:  logger,
	}
	gameEngine := game.NewEngine(context.Background(), config)

	uiGame, err := ui.NewUIGame(gameEngine, 0)
	if err != nil {
		log.Fatal(err)
	}

	ebiten.SetWindowSize(ui.ScreenWidth, ui.ScreenHeight)
	ebiten.SetWindowTitle("Generals RL UI")

	if err := ebiten.RunGame(uiGame); err != nil {
		log.Fatal(err)
	}
}
