package main

import (
	"context"
	"flag"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/config"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/ui"
	"github.com/rs/zerolog"
)

func main() {
	// Command line flags
	configPath := flag.String("config", "", "Path to config file")
	humanPlayer := flag.Int("human", -1, "Which player should be human controlled (0-based, -1 to use config default)")
	numPlayers := flag.Int("players", -1, "Number of players (2-4, -1 to use config default)")
	mapWidth := flag.Int("width", -1, "Map width (-1 to use config default)")
	mapHeight := flag.Int("height", -1, "Map height (-1 to use config default)")
	allAI := flag.Bool("ai-only", false, "Run with all AI players (original mode)")
	flag.Parse()

	// Initialize configuration
	if err := config.Init(*configPath); err != nil {
		log.Fatalf("Failed to initialize config: %v", err)
	}

	cfg := config.Get()

	// Use config defaults if not overridden by flags
	if *humanPlayer == -1 {
		*humanPlayer = cfg.UI.Defaults.HumanPlayer
	}
	if *numPlayers == -1 {
		*numPlayers = cfg.UI.Defaults.NumPlayers
	}
	if *mapWidth == -1 {
		*mapWidth = cfg.UI.Defaults.MapWidth
	}
	if *mapHeight == -1 {
		*mapHeight = cfg.UI.Defaults.MapHeight
	}
	// allAI flag overrides config if set to true
	if !*allAI {
		*allAI = cfg.UI.Defaults.AIOnly
	}

	// Validate inputs
	if *numPlayers < 2 || *numPlayers > 4 {
		log.Fatal("Number of players must be between 2 and 4")
	}
	if *humanPlayer >= *numPlayers {
		log.Fatal("Human player index must be less than number of players")
	}

	// Set up logging based on config
	logLevel, err := zerolog.ParseLevel(cfg.Server.GameServer.LogLevel)
	if err != nil {
		logLevel = zerolog.InfoLevel
	}

	var logger zerolog.Logger
	if cfg.Server.GameServer.LogFormat == "json" || os.Getenv("APP_ENV") == "production" {
		logger = zerolog.New(os.Stdout).Level(logLevel).With().Timestamp().Logger()
	} else {
		logger = zerolog.New(zerolog.ConsoleWriter{Out: os.Stdout}).Level(logLevel).With().Timestamp().Logger()
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	gameConfig := game.GameConfig{
		Width:   *mapWidth,
		Height:  *mapHeight,
		Players: *numPlayers,
		Rng:     rng,
		Logger:  logger,
	}
	gameEngine := game.NewEngine(context.Background(), gameConfig)

	ebiten.SetWindowSize(ui.ScreenWidth(), ui.ScreenHeight())
	ebiten.SetWindowTitle(cfg.UI.Window.Title)

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
