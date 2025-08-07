package experience

import (
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
)

// RewardConfig holds configurable reward values
type RewardConfig struct {
	WinGame         float32
	LoseGame        float32
	CaptureCity     float32
	LoseCity        float32
	CaptureGeneral  float32
	LoseGeneral     float32
	TerritoryGained float32
	TerritoryLost   float32
	ArmyGained      float32
	ArmyLost        float32
	ArmyAdvantage   float32 // Scaling factor for army advantage
}

// DefaultRewardConfig returns the default reward configuration
func DefaultRewardConfig() *RewardConfig {
	return &RewardConfig{
		WinGame:         1.0,
		LoseGame:        -1.0,
		CaptureCity:     0.1,
		LoseCity:        -0.1,
		CaptureGeneral:  0.5,
		LoseGeneral:     -0.5,
		TerritoryGained: 0.01,
		TerritoryLost:   -0.01,
		ArmyGained:      0.001,
		ArmyLost:        -0.001,
		ArmyAdvantage:   0.05, // Max reward/penalty for army advantage
	}
}

// CalculateReward computes the reward for a player given state transition
func CalculateReward(prevState, currState *game.GameState, playerID int) float32 {
	return CalculateRewardWithConfig(prevState, currState, playerID, DefaultRewardConfig())
}

// CalculateRewardWithConfig computes reward using custom configuration
func CalculateRewardWithConfig(prevState, currState *game.GameState, playerID int, config *RewardConfig) float32 {
	reward := float32(0.0)

	// Check for game-ending conditions first
	if currState.IsGameOver() {
		winner := currState.GetWinner()
		if winner == playerID {
			return config.WinGame
		} else if winner != -1 {
			return config.LoseGame
		}
	}

	// Calculate territory changes
	prevTerritory := countPlayerTerritory(prevState, playerID)
	currTerritory := countPlayerTerritory(currState, playerID)
	territoryDiff := currTerritory - prevTerritory
	reward += float32(territoryDiff) * config.TerritoryGained

	// Calculate army changes
	prevArmies := countPlayerArmies(prevState, playerID)
	currArmies := countPlayerArmies(currState, playerID)
	armyDiff := currArmies - prevArmies
	reward += float32(armyDiff) * config.ArmyGained

	// Check for city captures/losses
	citiesGained, citiesLost := countCityChanges(prevState, currState, playerID)
	reward += float32(citiesGained) * config.CaptureCity
	reward += float32(citiesLost) * config.LoseCity

	// Check for general captures/losses
	generalsGained, generalsLost := countGeneralChanges(prevState, currState, playerID)
	reward += float32(generalsGained) * config.CaptureGeneral
	reward += float32(generalsLost) * config.LoseGeneral

	// Calculate army advantage bonus
	armyAdvantage := calculateArmyAdvantage(currState, playerID)
	reward += armyAdvantage * config.ArmyAdvantage

	return reward
}

// countPlayerTerritory counts tiles owned by a player
func countPlayerTerritory(state *game.GameState, playerID int) int {
	count := 0
	for _, tile := range state.Board.T {
		if tile.Owner == playerID {
			count++
		}
	}
	return count
}

// countPlayerArmies counts total armies owned by a player
func countPlayerArmies(state *game.GameState, playerID int) int {
	count := 0
	for _, tile := range state.Board.T {
		if tile.Owner == playerID {
			count += tile.Army
		}
	}
	return count
}

// countCityChanges counts cities gained and lost
func countCityChanges(prevState, currState *game.GameState, playerID int) (gained, lost int) {
	for i, currTile := range currState.Board.T {
		if !currTile.IsCity() {
			continue
		}

		prevTile := &prevState.Board.T[i]

		// City gained
		if prevTile.Owner != playerID && currTile.Owner == playerID {
			gained++
		}

		// City lost
		if prevTile.Owner == playerID && currTile.Owner != playerID {
			lost++
		}
	}
	return gained, lost
}

// countGeneralChanges counts generals gained and lost
func countGeneralChanges(prevState, currState *game.GameState, playerID int) (gained, lost int) {
	for i, currTile := range currState.Board.T {
		if !currTile.IsGeneral() {
			continue
		}

		prevTile := &prevState.Board.T[i]

		// General captured (enemy general)
		if prevTile.Owner != playerID && prevTile.Owner >= 0 && currTile.Owner == playerID {
			gained++
		}

		// General lost (our general)
		if prevTile.Owner == playerID && currTile.Owner != playerID {
			lost++
		}
	}
	return gained, lost
}

// calculateArmyAdvantage returns normalized army advantage [-1, 1]
func calculateArmyAdvantage(state *game.GameState, playerID int) float32 {
	playerArmies := 0
	enemyArmies := 0

	for _, tile := range state.Board.T {
		if tile.Owner == playerID {
			playerArmies += tile.Army
		} else if tile.Owner >= 0 {
			enemyArmies += tile.Army
		}
	}

	// Avoid division by zero
	totalArmies := playerArmies + enemyArmies
	if totalArmies == 0 {
		return 0.0
	}

	// Calculate advantage as proportion of total armies, scaled to [-1, 1]
	advantage := float32(playerArmies-enemyArmies) / float32(totalArmies)
	return advantage
}

// CalculateStepReward calculates intermediate rewards during gameplay
func CalculateStepReward(state *game.GameState, playerID int, action *game.Action) float32 {
	// This could be extended to include action-specific rewards
	// For now, returns 0 as step rewards are calculated in state transitions
	return 0.0
}

// CalculatePotentialReward estimates potential reward for planning
func CalculatePotentialReward(state *game.GameState, playerID int, targetX, targetY int) float32 {
	reward := float32(0.0)
	config := DefaultRewardConfig()

	tileIdx := targetY*state.Board.W + targetX
	if tileIdx < 0 || tileIdx >= len(state.Board.T) {
		return 0.0
	}

	tile := &state.Board.T[tileIdx]

	// Potential city capture
	if tile.IsCity() && tile.Owner != playerID {
		reward += config.CaptureCity
	}

	// Potential general capture
	if tile.IsGeneral() && tile.Owner != playerID && tile.Owner >= 0 {
		reward += config.CaptureGeneral
	}

	// Territory gain
	if tile.Owner != playerID {
		reward += config.TerritoryGained
	}

	return reward
}

// NormalizeReward applies normalization to keep rewards in reasonable range
func NormalizeReward(reward float32) float32 {
	// Simple clipping for now
	if reward > 1.0 {
		return 1.0
	} else if reward < -1.0 {
		return -1.0
	}
	return reward
}
