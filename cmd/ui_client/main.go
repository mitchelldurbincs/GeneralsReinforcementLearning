package main

import (
	"context"
	"flag"
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
	// Command line flags
	humanPlayer := flag.Int("human", 0, "Which player should be human controlled (0-based)")
	numPlayers := flag.Int("players", 2, "Number of players (2-4)")
	mapWidth := flag.Int("width", 20, "Map width")
	mapHeight := flag.Int("height", 15, "Map height")
	allAI := flag.Bool("ai-only", false, "Run with all AI players (original mode)")
	flag.Parse()
	
	// Validate inputs
	if *numPlayers < 2 || *numPlayers > 4 {
		log.Fatal("Number of players must be between 2 and 4")
	}
	if *humanPlayer >= *numPlayers {
		log.Fatal("Human player index must be less than number of players")
	}
	
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()
	config := game.GameConfig{
		Width:   *mapWidth,
		Height:  *mapHeight,
		Players: *numPlayers,
		Rng:     rng,
		Logger:  logger,
	}
	gameEngine := game.NewEngine(context.Background(), config)
	
	ebiten.SetWindowSize(ui.ScreenWidth, ui.ScreenHeight)
	ebiten.SetWindowTitle("Generals RL UI")
	
	if *allAI {
		// Original AI-only mode
		uiGame, err := ui.NewUIGame(gameEngine, 0)
		if err != nil {
			log.Fatal(err)
		}
		if err := ebiten.RunGame(uiGame); err != nil {
			log.Fatal(err)
		}
	} else {
		// New human player mode
		playerConfigs := make([]ui.PlayerConfig, *numPlayers)
		for i := 0; i < *numPlayers; i++ {
			if i == *humanPlayer {
				playerConfigs[i] = ui.PlayerConfig{
					Type: ui.PlayerTypeHuman,
					ID:   i,
				}
			} else {
				playerConfigs[i] = ui.PlayerConfig{
					Type: ui.PlayerTypeAI,
					ID:   i,
				}
			}
		}
		
		humanGame, err := ui.NewHumanGame(gameEngine, playerConfigs)
		if err != nil {
			log.Fatal(err)
		}
		
		if err := ebiten.RunGame(humanGame); err != nil {
			log.Fatal(err)
		}
	}
}