package main

import (
	"context" // Import the context package
	"errors"  // Import the errors package for errors.Is
	"math/rand"
	"os"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log" // Using the global logger
)

func main() {
	// --- Zerolog Configuration ---
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	log.Logger = log.Output(zerolog.ConsoleWriter{
		Out:        os.Stderr,
		TimeFormat: time.RFC3339,
		NoColor:    false,
	}).With().Timestamp().Caller().Logger()

	// Example for future production setup:
	// if os.Getenv("APP_ENV") == "production" {
	// 	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	// 	log.Logger = zerolog.New(os.Stdout).With().Timestamp().Logger()
	// }

	log.Info().Msg("Logger initialized")

	randomActionDemo()
	// manualTest()
}

func randomActionDemo() {
	// --- Context for the demo ---
	// For this standalone demo, context.Background() is sufficient.
	// It's an empty context, signifying no specific deadline or cancellation signal from a parent.
	// In a server environment, this context would typically come from the incoming request.
	ctx := context.Background()

	seed := time.Now().UnixNano()
	log.Info().Int64("seed", seed).Msg("Starting random action demo")
	rng := rand.New(rand.NewSource(seed))

	// Create a 8x8 game with 2 players
	// Pass the context and the global logger (log.Logger) to NewEngine
	config := game.GameConfig{
		Width:   8,
		Height:  8,
		Players: 2,
		Rng:     rng,
		Logger:  log.Logger,
	}
	g := game.NewEngine(ctx, config)
	if g == nil {
		log.Fatal().Msg("Failed to create game engine (NewEngine returned nil, possibly due to context cancellation during init)")
		return
	}


	log.Info().Msgf("Initial board:\n%s", g.Board())
	initialPlayers := g.GameState().Players
	for i, p := range initialPlayers {
		log.Info().
			Int("player_id", p.ID).
			Int("initial_army_count", p.ArmyCount).
			Bool("initial_alive", p.Alive).
			Int("initial_general_idx", p.GeneralIdx).
			Msgf("Initial stats for player %d", i)
	}

	maxTurns := 50
	for turn := 0; turn < maxTurns && !g.IsGameOver(); turn++ {
		actions := game.GenerateRandomActions(g, rng) // g is *game.Engine

		turnLogger := log.With().Int("turn", turn+1).Logger()

		// Pass the context to the Step method
		if err := g.Step(ctx, actions); err != nil {
			// Check if the error is due to context cancellation/deadline
			if errors.Is(err, context.Canceled) {
				turnLogger.Warn().Err(err).Msg("Game step was canceled")
			} else if errors.Is(err, context.DeadlineExceeded) {
				turnLogger.Error().Err(err).Msg("Game step timed out")
			} else {
				turnLogger.Error().Err(err).Msg("Error processing game step")
			}
			break // Stop simulation on any error
		}

		if turn%5 == 0 || len(actions) > 0 {
			turnLogger.Info().Int("actions_count", len(actions)).Msg("Turn processed")

			if len(actions) > 0 {
				for _, action := range actions {
					if moveAction, ok := action.(*core.MoveAction); ok {
						turnLogger.Debug().
							Int("player_id", moveAction.PlayerID).
							Int("from_x", moveAction.FromX).
							Int("from_y", moveAction.FromY).
							Int("to_x", moveAction.ToX).
							Int("to_y", moveAction.ToY).
							Bool("move_all", moveAction.MoveAll).
							Msg("Player move action")
					}
				}
			}
			turnLogger.Info().Msgf("Board state:\n%s", g.Board())

			state := g.GameState()
			for _, player := range state.Players {
				turnLogger.Info().
					Int("player_id", player.ID).
					Int("army_count", player.ArmyCount).
					Bool("alive", player.Alive).
					Msg("Player status")
			}
		}
	}

	if g.IsGameOver() {
		winner := g.GetWinner()
		if winner >= 0 {
			log.Info().Int("winner_player_id", winner).Msg("🎉 Game Over! Player wins!")
		} else {
			log.Info().Msg("Game Over! No winner (draw or error).")
		}
	} else {
		log.Info().Int("max_turns_reached", maxTurns).Msg("Game reached maximum turns")
	}

	log.Info().Msgf("Final board:\n%s", g.Board())
}

