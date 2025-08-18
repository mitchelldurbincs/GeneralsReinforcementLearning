package gameserver

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// TestMaxGamesLimit tests that the server correctly enforces the max games limit
func TestMaxGamesLimit(t *testing.T) {
	// Create a game manager with a limit of 3 games
	maxGames := 3
	gm := NewGameManager(maxGames)

	// Create games up to the limit
	for i := 0; i < maxGames; i++ {
		game, id, err := gm.CreateGame(&gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
		})
		require.NoError(t, err, "Should be able to create game %d", i+1)
		require.NotNil(t, game)
		require.NotEmpty(t, id)
	}

	// Verify we have maxGames active
	assert.Equal(t, maxGames, gm.GetActiveGames())

	// Try to create one more game - should fail
	game, _, err := gm.CreateGame(&gamev1.GameConfig{
		Width:      10,
		Height:     10,
		MaxPlayers: 2,
	})
	assert.Error(t, err, "Should not be able to create game beyond limit")
	assert.Nil(t, game)
	assert.Contains(t, err.Error(), "server at capacity")

	// Active games should still be maxGames
	assert.Equal(t, maxGames, gm.GetActiveGames())
}

// TestMaxGamesZeroMeansUnlimited tests that maxGames=0 means unlimited
func TestMaxGamesZeroMeansUnlimited(t *testing.T) {
	// Create a game manager with no limit (maxGames = 0)
	gm := NewGameManager(0)

	// Create many games
	numGames := 20
	for i := 0; i < numGames; i++ {
		game, id, err := gm.CreateGame(&gamev1.GameConfig{
			Width:      10,
			Height:     10,
			MaxPlayers: 2,
		})
		require.NoError(t, err, "Should be able to create game %d", i+1)
		require.NotNil(t, game)
		require.NotEmpty(t, id)
	}

	// All games should be created successfully
	assert.Equal(t, numGames, gm.GetActiveGames())
}
