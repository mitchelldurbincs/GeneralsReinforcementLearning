package game

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPlayer_InitialState(t *testing.T) {
	player := Player{
		ID:         0,
		Alive:      true,
		ArmyCount:  1,
		GeneralIdx: 12,
		OwnedTiles: []int{12},
	}

	assert.Equal(t, 0, player.ID)
	assert.True(t, player.Alive)
	assert.Equal(t, 1, player.ArmyCount)
	assert.Equal(t, 12, player.GeneralIdx)
	assert.Equal(t, []int{12}, player.OwnedTiles)
}

func TestPlayer_Elimination(t *testing.T) {
	tests := []struct {
		name         string
		player       Player
		isEliminated bool
	}{
		{
			name: "alive player with general",
			player: Player{
				ID:         0,
				Alive:      true,
				GeneralIdx: 5,
				ArmyCount:  10,
				OwnedTiles: []int{5, 6, 7},
			},
			isEliminated: false,
		},
		{
			name: "eliminated player",
			player: Player{
				ID:         1,
				Alive:      false,
				GeneralIdx: -1,
				ArmyCount:  0,
				OwnedTiles: []int{},
			},
			isEliminated: true,
		},
		{
			name: "player marked alive but no general",
			player: Player{
				ID:         2,
				Alive:      true,
				GeneralIdx: -1,
				ArmyCount:  5,
				OwnedTiles: []int{1, 2},
			},
			isEliminated: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isEliminated := tt.player.GeneralIdx == -1
			assert.Equal(t, tt.isEliminated, isEliminated)
		})
	}
}

func TestPlayer_OwnedTiles(t *testing.T) {
	t.Run("add tiles", func(t *testing.T) {
		player := Player{
			ID:         0,
			Alive:      true,
			OwnedTiles: []int{},
		}

		// Simulate adding tiles
		player.OwnedTiles = append(player.OwnedTiles, 5)
		player.OwnedTiles = append(player.OwnedTiles, 10)
		player.OwnedTiles = append(player.OwnedTiles, 15)

		assert.Len(t, player.OwnedTiles, 3)
		assert.Contains(t, player.OwnedTiles, 5)
		assert.Contains(t, player.OwnedTiles, 10)
		assert.Contains(t, player.OwnedTiles, 15)
	})

	t.Run("remove tiles", func(t *testing.T) {
		player := Player{
			ID:         0,
			Alive:      true,
			OwnedTiles: []int{1, 2, 3, 4, 5},
		}

		// Simulate removing a tile (e.g., tile 3)
		newOwnedTiles := []int{}
		for _, tile := range player.OwnedTiles {
			if tile != 3 {
				newOwnedTiles = append(newOwnedTiles, tile)
			}
		}
		player.OwnedTiles = newOwnedTiles

		assert.Len(t, player.OwnedTiles, 4)
		assert.NotContains(t, player.OwnedTiles, 3)
	})
}

func TestPlayer_ArmyCount(t *testing.T) {
	tests := []struct {
		name     string
		player   Player
		expected int
	}{
		{
			name: "player with armies",
			player: Player{
				ID:        0,
				Alive:     true,
				ArmyCount: 50,
			},
			expected: 50,
		},
		{
			name: "eliminated player",
			player: Player{
				ID:        1,
				Alive:     false,
				ArmyCount: 0,
			},
			expected: 0,
		},
		{
			name: "new player",
			player: Player{
				ID:        2,
				Alive:     true,
				ArmyCount: 1,
			},
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.player.ArmyCount)
		})
	}
}

func TestPlayer_MultiplePlayersInteraction(t *testing.T) {
	// Test scenario with multiple players
	players := []Player{
		{
			ID:         0,
			Alive:      true,
			ArmyCount:  25,
			GeneralIdx: 0,
			OwnedTiles: []int{0, 1, 2, 3, 4},
		},
		{
			ID:         1,
			Alive:      true,
			ArmyCount:  30,
			GeneralIdx: 20,
			OwnedTiles: []int{20, 21, 22, 23, 24, 25},
		},
		{
			ID:         2,
			Alive:      false,
			ArmyCount:  0,
			GeneralIdx: -1,
			OwnedTiles: []int{},
		},
	}

	// Count alive players
	aliveCount := 0
	for _, p := range players {
		if p.Alive {
			aliveCount++
		}
	}
	assert.Equal(t, 2, aliveCount)

	// Total army count
	totalArmy := 0
	for _, p := range players {
		totalArmy += p.ArmyCount
	}
	assert.Equal(t, 55, totalArmy)

	// Total owned tiles
	totalTiles := 0
	for _, p := range players {
		totalTiles += len(p.OwnedTiles)
	}
	assert.Equal(t, 11, totalTiles)
}
