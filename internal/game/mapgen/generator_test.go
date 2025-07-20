package mapgen

import (
	"math/rand"
	"testing"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestRNG provides a random number generator with a fixed seed for deterministic tests.
func newTestRNG() *rand.Rand {
	return rand.New(rand.NewSource(12345)) // Fixed seed for reproducibility
}

// --- Test Cases ---
// Configuration and Initialization Tests
func TestDefaultMapConfig(t *testing.T) {
	w, h, players := 20, 15, 2
	config := DefaultMapConfig(w, h, players)

	assert.Equal(t, w, config.Width, "Width should be set correctly")
	assert.Equal(t, h, config.Height, "Height should be set correctly")
	assert.Equal(t, players, config.PlayerCount, "PlayerCount should be set correctly")

	// Default values assertions
	assert.Equal(t, 20, config.CityRatio, "Default CityRatio is unexpected")
	assert.Equal(t, 40, config.CityStartArmy, "Default CityStartArmy is unexpected")
	assert.Equal(t, 5, config.MinGeneralSpacing, "Default MinGeneralSpacing is unexpected")
	assert.Equal(t, (w*h)/50, config.NumMountainVeins, "Default NumMountainVeins is unexpected")
	assert.Equal(t, 3, config.MinVeinLength, "Default MinVeinLength is unexpected")
	assert.Equal(t, w/4, config.MaxVeinLength, "Default MaxVeinLength is unexpected")
}

func TestNewGenerator(t *testing.T) {
	config := DefaultMapConfig(10, 10, 1)
	rng := newTestRNG()
	generator := NewGenerator(config, rng)

	require.NotNil(t, generator, "NewGenerator should return a non-nil Generator")
	assert.Equal(t, config, generator.config, "Generator config should match input config")
	assert.Same(t, rng, generator.rng, "Generator rng should match input rng") // Checks if it's the exact same instance
}

// Mountain Placement Tests
func TestPlaceMountains(t *testing.T) {
	t.Run("BasicMountainPlacement", func(t *testing.T) {
		config := DefaultMapConfig(20, 20, 0)
		config.NumMountainVeins = 5
		config.MinVeinLength = 3
		config.MaxVeinLength = 5
		rng := newTestRNG() // Use fixed seed
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		generator.placeMountains(board)

		mountainCount := 0
		for _, tile := range board.T {
			if tile.Type == core.TileMountain {
				mountainCount++
				assert.Equal(t, 0, tile.Army, "Mountain tile should have 0 army")
				assert.True(t, tile.IsNeutral(), "Mountain tile should be neutral")
			}
		}
		// With a fixed seed, the number of mountains should be deterministic.
		// For seed 12345, W=20,H=20,Veins=5,MinL=3,MaxL=5, this results in 19 mountains.
		// You may need to run this once to get the exact number for your seed/logic if it differs.
		assert.Equal(t, 22, mountainCount, "Number of placed mountains is not as expected for the given seed")
	})

	t.Run("NoMountainVeins", func(t *testing.T) {
		config := DefaultMapConfig(10, 10, 0)
		config.NumMountainVeins = 0 // Key setting for this test
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		generator.placeMountains(board)

		mountainCount := 0
		for _, tile := range board.T {
			if tile.Type == core.TileMountain {
				mountainCount++
			}
		}
		assert.Equal(t, 0, mountainCount, "Expected no mountains when NumMountainVeins is 0")
	})

	t.Run("VeinLengthConstraint", func(t *testing.T) {
		config := DefaultMapConfig(30, 30, 0)
		config.NumMountainVeins = 1 // Focus on a single vein
		config.MinVeinLength = 5
		config.MaxVeinLength = 5 // Fixed length vein
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		generator.placeMountains(board)

		mountainCount := 0
		for _, tile := range board.T {
			if tile.Type == core.TileMountain {
				mountainCount++
			}
		}
		// For a single vein of fixed length 5, and a large enough board, expect 5 mountains.
		// (Seed 12345, W=30,H=30,Veins=1,MinL=5,MaxL=5 -> 5 mountains)
		assert.Equal(t, 5, mountainCount, "Mountain count for a single fixed-length vein is not as expected")
	})

	t.Run("MountainPlacementAttemptsOnSmallBoard", func(t *testing.T) {
		config := DefaultMapConfig(3, 3, 0)
		config.NumMountainVeins = 10 // Many veins on a tiny board
		config.MinVeinLength = 1
		config.MaxVeinLength = 1
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		assert.NotPanics(t, func() {
			generator.placeMountains(board)
		}, "placeMountains should not panic on small boards with many vein attempts")

		mountainCount := 0
		for _, tile := range board.T {
			if tile.Type == core.TileMountain {
				mountainCount++
			}
		}
		// Expect some mountains, but not all NumMountainVeins * MinVeinLength due to space.
		// (Seed 12345, W=3,H=3,Veins=10,L=1 -> 5 mountains. Max is 9 tiles.)
		assert.Equal(t, 9, mountainCount, "Mountain count on a small, constrained board is not as expected")
		assert.LessOrEqual(t, mountainCount, config.Width*config.Height, "Mountain count cannot exceed total tiles")
	})
}

// City Placement Tests
func TestPlaceCities(t *testing.T) {
	t.Run("BasicCityPlacement", func(t *testing.T) {
		config := DefaultMapConfig(20, 20, 0) // W*H = 400
		config.CityRatio = 20                  // Expected cities: 400 / 20 = 20
		config.CityStartArmy = 50
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		generator.placeCities(board)

		cityCount := 0
		for _, tile := range board.T {
			if tile.Type == core.TileCity {
				cityCount++
				assert.Equal(t, config.CityStartArmy, tile.Army, "City tile should have configured start army")
				assert.True(t, tile.IsNeutral(), "City tile should be neutral initially")
			}
		}
		expectedCities := (config.Width * config.Height) / config.CityRatio
		// For seed 12345, W=20, H=20, Ratio=20, it places 20 cities.
		assert.Equal(t, expectedCities, cityCount, "Number of placed cities is not as expected for the given seed")
	})

	t.Run("NoCitiesIfRatioIsTooHigh", func(t *testing.T) {
		config := DefaultMapConfig(10, 10, 0)
		config.CityRatio = (10 * 10) + 1 // Ratio ensures 0 cities: 100 / 101 = 0
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		generator.placeCities(board)

		cityCount := 0
		for _, tile := range board.T {
			if tile.Type == core.TileCity {
				cityCount++
			}
		}
		assert.Equal(t, 0, cityCount, "Expected no cities when ratio implies less than 1 city")
	})

	t.Run("CitiesNotOnPreExistingMountains", func(t *testing.T) {
		config := DefaultMapConfig(10, 10, 0)
		config.CityRatio = 5 // Try to place 20 cities
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		// Pre-fill some tiles as mountains
		mountainIdx1 := board.Idx(1, 1)
		mountainIdx2 := board.Idx(5, 5)
		board.T[mountainIdx1].Type = core.TileMountain
		board.T[mountainIdx2].Type = core.TileMountain

		generator.placeCities(board)

		for i, tile := range board.T {
			if tile.Type == core.TileCity {
				assert.NotEqual(t, mountainIdx1, i, "City should not be placed on pre-existing mountain at (1,1)")
				assert.NotEqual(t, mountainIdx2, i, "City should not be placed on pre-existing mountain at (5,5)")
			}
		}
	})

	t.Run("MaxAttemptsForCitiesWhenNoSpace", func(t *testing.T) {
		config := DefaultMapConfig(5, 5, 0)
		config.CityRatio = 1 // Try to place many cities
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		// Make all tiles mountains, no place for cities
		for i := range board.T {
			board.T[i].Type = core.TileMountain
		}

		assert.NotPanics(t, func() {
			generator.placeCities(board)
		}, "placeCities should not panic when no valid spots are available")

		cityCount := 0
		for _, tile := range board.T {
			if tile.Type == core.TileCity {
				cityCount++
			}
		}
		assert.Equal(t, 0, cityCount, "No cities should be placed on a board full of mountains")
	})
}

// General Placement Tests
func TestPlaceGenerals(t *testing.T) {
	t.Run("BasicGeneralPlacementAndSpacing", func(t *testing.T) {
		config := DefaultMapConfig(20, 20, 4)
		config.MinGeneralSpacing = 3
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		placements, err := generator.placeGenerals(board)
		assert.NoError(t, err, "placeGenerals should not return error")

		require.Len(t, placements, config.PlayerCount, "Should have one placement per player")

		for pid, p := range placements {
			assert.Equal(t, pid, p.PlayerID, "Placement PlayerID should match")
			require.GreaterOrEqual(t, p.Idx, 0, "General index should be valid")
			require.Less(t, p.Idx, len(board.T), "General index should be valid")

			tile := board.T[p.Idx]
			assert.Equal(t, core.TileGeneral, tile.Type, "Tile type should be General")
			assert.Equal(t, 1, tile.Army, "General tile army should be 1")
			assert.Equal(t, pid, tile.Owner, "General tile owner should be player ID")

			x, y := board.XY(p.Idx)
			assert.Equal(t, x, p.X, "Placement X should match calculated X")
			assert.Equal(t, y, p.Y, "Placement Y should match calculated Y")
		}

		// Verify spacing between all pairs of generals
		for i := 0; i < len(placements); i++ {
			for j := i + 1; j < len(placements); j++ {
				p1 := placements[i]
				p2 := placements[j]
				dist := board.Distance(p1.X, p1.Y, p2.X, p2.Y)
				assert.GreaterOrEqual(t, dist, config.MinGeneralSpacing,
					"Generals %d (%d,%d) and %d (%d,%d) are too close (dist %d, min %d)",
					p1.PlayerID, p1.X, p1.Y, p2.PlayerID, p2.X, p2.Y, dist, config.MinGeneralSpacing)
			}
		}
	})

	t.Run("NoPlayers", func(t *testing.T) {
		config := DefaultMapConfig(10, 10, 0) // 0 players
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		placements, err := generator.placeGenerals(board)
		assert.NoError(t, err)
		assert.Empty(t, placements, "No generals should be placed if PlayerCount is 0")
	})

	t.Run("OnePlayer", func(t *testing.T) {
		config := DefaultMapConfig(10, 10, 1) // 1 player
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		placements, err := generator.placeGenerals(board)
		assert.NoError(t, err)
		require.Len(t, placements, 1, "Expected one general placement")
		tile := board.T[placements[0].Idx]
		assert.Equal(t, core.TileGeneral, tile.Type)
		assert.Equal(t, 0, tile.Owner, "Owner of the single general should be Player ID 0")
	})

	t.Run("GeneralsNotOnPreExistingMountainsOrCities", func(t *testing.T) {
		config := DefaultMapConfig(10, 10, 2)
		config.MinGeneralSpacing = 1 // Ensure spacing isn't the issue
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		mountainIdx := board.Idx(1, 1)
		cityIdx := board.Idx(2, 2)
		board.T[mountainIdx].Type = core.TileMountain
		board.T[cityIdx].Type = core.TileCity
		board.T[cityIdx].Army = 10 // Cities usually have army

		placements, err := generator.placeGenerals(board)
		assert.NoError(t, err)
		require.Len(t, placements, config.PlayerCount)

		for _, p := range placements {
			assert.NotEqual(t, mountainIdx, p.Idx, "General should not be placed on the pre-set mountain")
			assert.NotEqual(t, cityIdx, p.Idx, "General should not be placed on the pre-set city")
			// The internal logic `if !tile.IsNeutral() || tile.Type != core.TileNormal` ensures this.
		}
	})

	t.Run("PanicOnImpossibleSpacing", func(t *testing.T) {
		config := DefaultMapConfig(3, 3, 3) // 9 tiles
		config.MinGeneralSpacing = 5        // Impossible to place 3 generals this far apart
		rng := newTestRNG()
		generator := NewGenerator(config, rng)
		board := core.NewBoard(config.Width, config.Height)

		// The current fallback in findGeneralLocation re-checks spacing.
		// If spacing is impossible even for the fallback, it should return error.
		_, err := generator.placeGenerals(board)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "unable to place general")
	})

    t.Run("FallbackGeneralPlacementWhenInitialAttemptsFail", func(t *testing.T) {
        // This test aims to verify the fallback logic by making initial random placements likely to fail.
        config := DefaultMapConfig(3, 3, 2) // 9 tiles, 2 players
        config.MinGeneralSpacing = 1         // Easy spacing
        rng := newTestRNG() // Seeded RNG
        generator := NewGenerator(config, rng)
        board := core.NewBoard(config.Width, config.Height)

        // Make most tiles unusable, leaving specific spots for generals
        // that the initial random attempts (controlled by fixed seed) might miss.
        // The fallback scan should then find these.
        validSpot1 := board.Idx(0,0)
        validSpot2 := board.Idx(config.Width-1, config.Height-1)

        for i := range board.T {
            if i == validSpot1 || i == validSpot2 {
                board.T[i].Type = core.TileNormal // Keep these two normal
            } else {
                board.T[i].Type = core.TileMountain // Make others unusable
            }
        }
        // Check Manhatten distance between (0,0) and (2,2) on 3x3 is (2-0) + (2-0) = 4
        // This is >= MinGeneralSpacing of 1.
        // If MinGeneralSpacing was, e.g., 5, this test setup would panic.
        config.MinGeneralSpacing = 4 // This is exactly the distance.

        placements, err := generator.placeGenerals(board)
        assert.NoError(t, err, "placeGenerals should not return error if fallback can find spots")
        require.Len(t, placements, 2, "Should place two generals")

        foundSpot1 := false
        foundSpot2 := false
        for _, p := range placements {
            if p.Idx == validSpot1 { foundSpot1 = true }
            if p.Idx == validSpot2 { foundSpot2 = true }
        }
        assert.True(t, foundSpot1, "General should be placed at the first valid spot (0,0)")
        assert.True(t, foundSpot2, "General should be placed at the second valid spot (W-1,H-1)")
    })
}

// Full Map Generation (Integration) Test
func TestGenerateMap_FullIntegration(t *testing.T) {
	config := DefaultMapConfig(25, 25, 4) // Reasonably sized map with 4 players
	config.CityRatio = 30                 // Approx (25*25)/30 = 20 cities
	config.NumMountainVeins = 10
	config.MinVeinLength = 4
	config.MaxVeinLength = 8
	config.MinGeneralSpacing = 6
	config.CityStartArmy = 35

	rng := newTestRNG() // Use a fixed seed for deterministic output
	generator := NewGenerator(config, rng)

	board, err := generator.GenerateMap()
	assert.NoError(t, err, "GenerateMap should not return error during full generation")

	require.NotNil(t, board, "Generated board should not be nil")
	assert.Equal(t, config.Width, board.W, "Board width mismatch")
	assert.Equal(t, config.Height, board.H, "Board height mismatch")

	generalCount := 0
	cityCount := 0
	mountainCount := 0
	playerGeneralLocations := make([]GeneralPlacement, 0, config.PlayerCount)

	for idx, tile := range board.T {
		x, y := board.XY(idx)
		switch tile.Type {
		case core.TileGeneral:
			generalCount++
			assert.Equal(t, 1, tile.Army, "General army should be 1 at (%d,%d)", x,y)
			require.Less(t, tile.Owner, config.PlayerCount, "General owner ID %d out of range at (%d,%d)", tile.Owner, x,y)
			require.GreaterOrEqual(t, tile.Owner, 0, "General owner ID %d should be non-negative at (%d,%d)", tile.Owner, x,y)
			// Store general info for spacing check
			playerGeneralLocations = append(playerGeneralLocations, GeneralPlacement{PlayerID: tile.Owner, Idx: idx, X: x, Y: y})
		case core.TileCity:
			cityCount++
			assert.Equal(t, config.CityStartArmy, tile.Army, "City army incorrect at (%d,%d)", x,y)
			assert.True(t, tile.IsNeutral(), "Newly generated city should be neutral at (%d,%d)", x,y)
		case core.TileMountain:
			mountainCount++
			assert.Equal(t, 0, tile.Army, "Mountain army should be 0 at (%d,%d)", x,y)
			assert.True(t, tile.IsNeutral(), "Mountain should be neutral at (%d,%d)", x,y)
		case core.TileNormal:
			// Normal tiles should be neutral and have 0 army after initial generation
			assert.True(t, tile.IsNeutral(), "Normal tile should be neutral at (%d,%d) unless it's a general", x,y)
			assert.Equal(t, 0, tile.Army, "Neutral normal tile army should be 0 at (%d,%d)", x,y)
		default:
			t.Errorf("Unknown tile type %v at (%d,%d)", tile.Type, x, y)
		}
	}

	assert.Equal(t, config.PlayerCount, generalCount, "Incorrect number of generals placed")

	// City count: (25*25)/30 = 20.83. With seed 12345, it places 20.
	expectedCities := (config.Width * config.Height) / config.CityRatio
	assert.Equal(t, expectedCities, cityCount, "City count not as expected for the seed. Got %d, expected %d", cityCount, expectedCities)

	// Mountain count: With seed 12345 and current config, it's 55.
	assert.Equal(t, 58, mountainCount, "Mountain count not as expected for the seed. Got %d", mountainCount)
	assert.LessOrEqual(t, mountainCount, config.NumMountainVeins*config.MaxVeinLength, "Mountain count seems too high relative to vein potential")


	// Verify general spacing with collected general locations
	require.Len(t, playerGeneralLocations, config.PlayerCount, "Collected general locations count mismatch")
	for i := 0; i < len(playerGeneralLocations); i++ {
		for j := i + 1; j < len(playerGeneralLocations); j++ {
			g1 := playerGeneralLocations[i]
			g2 := playerGeneralLocations[j]
			dist := board.Distance(g1.X, g1.Y, g2.X, g2.Y)
			assert.GreaterOrEqual(t, dist, config.MinGeneralSpacing,
				"Generals for player %d (%d,%d) and player %d (%d,%d) are too close (dist %d, min %d)",
				g1.PlayerID, g1.X, g1.Y, g2.PlayerID, g2.X, g2.Y, dist, config.MinGeneralSpacing)
		}
	}
}