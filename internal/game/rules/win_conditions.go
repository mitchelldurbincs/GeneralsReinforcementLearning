package rules

import "github.com/rs/zerolog"

// WinConditionChecker handles game over detection and winner determination
type WinConditionChecker struct {
	logger         zerolog.Logger
	originalPlayers int
}

// NewWinConditionChecker creates a new win condition checker
func NewWinConditionChecker(logger zerolog.Logger, originalPlayers int) *WinConditionChecker {
	return &WinConditionChecker{
		logger:          logger.With().Str("component", "WinConditionChecker").Logger(),
		originalPlayers: originalPlayers,
	}
}

// CheckGameOver determines if the game is over based on the number of alive players
// Returns (isGameOver, winnerID)
func (wc *WinConditionChecker) CheckGameOver(players []Player) (bool, int) {
	wc.logger.Debug().Msg("Checking game over conditions")
	aliveCount := 0
	var alivePlayers []int
	var lastAliveID int
	
	for _, p := range players {
		if p.IsAlive() {
			aliveCount++
			playerID := p.GetID()
			alivePlayers = append(alivePlayers, playerID)
			lastAliveID = playerID
		}
	}

	// Game is over only if:
	// - 0 players alive (draw)
	// - 1 player alive AND there were originally more than 1 player
	var gameOver bool
	if wc.originalPlayers > 1 {
		gameOver = aliveCount <= 1
	} else {
		gameOver = aliveCount == 0
	}

	var winnerID int = -1
	if gameOver && aliveCount == 1 {
		winnerID = lastAliveID
		wc.logger.Info().Int("winner_player_id", winnerID).Msg("Winner determined")
	} else if gameOver {
		wc.logger.Info().Msg("No winner found (draw, or all players eliminated simultaneously)")
	}

	wc.logger.Debug().Bool("is_game_over", gameOver).Int("alive_player_count", aliveCount).Interface("alive_players_ids", alivePlayers).Msg("Game over check complete")
	
	return gameOver, winnerID
}

// Player interface to avoid circular imports
type Player interface {
	GetID() int
	IsAlive() bool
}