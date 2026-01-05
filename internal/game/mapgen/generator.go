package mapgen

import (
	"fmt"
	"math/rand"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/config"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

// MapConfig holds configuration for map generation
type MapConfig struct {
	Width             int
	Height            int
	PlayerCount       int
	CityRatio         int // 1 city per N tiles
	CityStartArmy     int
	MinGeneralSpacing int
	NumMountainVeins  int // Number of mountain veins/ranges to generate
	MinVeinLength     int // Minimum length of a mountain vein
	MaxVeinLength     int // Maximum length of a mountain vein
}

// DefaultMapConfig returns a sensible default configuration
func DefaultMapConfig(w, h, players int) MapConfig {
	cfg := config.Get()
	return MapConfig{
		Width:             w,
		Height:            h,
		PlayerCount:       players,
		CityRatio:         cfg.Game.Map.CityRatio,
		CityStartArmy:     cfg.Game.Map.CityStartArmy,
		MinGeneralSpacing: cfg.Game.Map.MinGeneralSpacing,
		NumMountainVeins:  (w * h) / 50, // Example: 1 vein per 100 tiles
		MinVeinLength:     3,
		MaxVeinLength:     w / 4, // Example: Max vein length is a quarter of the width
	}
}

// Generator handles map generation with deterministic RNG
type Generator struct {
	config MapConfig
	rng    *rand.Rand
}

// NewGenerator creates a new map generator
func NewGenerator(config MapConfig, rng *rand.Rand) *Generator {
	return &Generator{
		config: config,
		rng:    rng,
	}
}

// GenerateMap creates a new board with mountains, cities, and generals placed
func (g *Generator) GenerateMap() (*core.Board, error) {
	board := core.NewBoard(g.config.Width, g.config.Height)

	g.placeMountains(board) // Place mountains first
	g.placeCities(board)
	_, err := g.placeGenerals(board)
	if err != nil {
		return nil, fmt.Errorf("failed to generate map (%dx%d, %d players): %w", g.config.Width, g.config.Height, g.config.PlayerCount, err)
	}

	return board, nil
}

func (g *Generator) placeMountains(b *core.Board) {
	for range g.config.NumMountainVeins {
		// Attempt to find a starting point for the vein
		startX, startY := -1, -1
		maxSeedAttempts := 100 // Try to find a valid seed for the vein start
		foundSeed := false
		for range maxSeedAttempts {
			sx, sy := g.rng.Intn(b.W), g.rng.Intn(b.H)
			sIdx := b.Idx(sx, sy)
			// Seed must be on a normal, neutral tile
			if b.T[sIdx].Type == core.TileNormal && b.T[sIdx].IsNeutral() {
				startX, startY = sx, sy
				foundSeed = true
				break
			}
		}

		if !foundSeed {
			continue // Could not find a suitable seed for this vein, try next vein
		}

		currentX, currentY := startX, startY
		idx := b.Idx(currentX, currentY)
		b.T[idx].Type = core.TileMountain
		b.T[idx].Army = 0 // Mountains are impassable and have no army
		// Owner remains NeutralID (default)

		veinLength := g.config.MinVeinLength
		if g.config.MaxVeinLength > g.config.MinVeinLength {
			veinLength += g.rng.Intn(g.config.MaxVeinLength - g.config.MinVeinLength + 1)
		}

		for i := 1; i < veinLength; i++ {
			// Get valid neighbors (N, S, E, W) that are normal tiles
			potentialNextSteps := []struct{ x, y int }{}
			// Directions: N, E, S, W
			dx := []int{0, 1, 0, -1}
			dy := []int{-1, 0, 1, 0} // Common graphics coordinates: (0,0) top-left

			// Shuffle directions to make veins more random
			g.rng.Shuffle(len(dx), func(i, j int) { dx[i], dx[j] = dx[j], dx[i]; dy[i], dy[j] = dy[j], dy[i] })

			for j := range 4 {
				nx, ny := currentX+dx[j], currentY+dy[j]
				if nx >= 0 && nx < b.W && ny >= 0 && ny < b.H {
					nIdx := b.Idx(nx, ny)
					// Vein can only grow into normal, neutral tiles
					if b.T[nIdx].Type == core.TileNormal && b.T[nIdx].IsNeutral() {
						potentialNextSteps = append(potentialNextSteps, struct{ x, y int }{nx, ny})
					}
				}
			}

			if len(potentialNextSteps) == 0 {
				break // Cannot extend vein further from this point
			}

			// Pick a random valid direction to grow
			nextStep := potentialNextSteps[g.rng.Intn(len(potentialNextSteps))]
			currentX, currentY = nextStep.x, nextStep.y
			idx = b.Idx(currentX, currentY)
			b.T[idx].Type = core.TileMountain
			b.T[idx].Army = 0
		}
	}
}

func (g *Generator) placeCities(b *core.Board) {
	want := (b.W * b.H) / g.config.CityRatio
	placed := 0

	maxAttempts := want * 20 // Increased attempts slightly
	attempts := 0

	for placed < want && attempts < maxAttempts {
		x, y := g.rng.Intn(b.W), g.rng.Intn(b.H)
		idx := b.Idx(x, y)
		t := &b.T[idx]

		// Cities can only be placed on neutral, normal tiles
		if t.IsNeutral() && t.Type == core.TileNormal {
			t.Type = core.TileCity
			t.Army = g.config.CityStartArmy
			placed++
		}
		attempts++
	}
}

func (g *Generator) placeGenerals(b *core.Board) ([]GeneralPlacement, error) {
	placements := make([]GeneralPlacement, g.config.PlayerCount)

	for pid := range g.config.PlayerCount {
		placement, err := g.findGeneralLocation(b, placements[:pid])
		if err != nil {
			return nil, fmt.Errorf("failed to place general for player %d on %dx%d map (spacing: %d): %w", pid, g.config.Width, g.config.Height, g.config.MinGeneralSpacing, err)
		}

		t := &b.T[placement.Idx]
		t.Owner = pid
		t.Army = 2  // Start with 2 armies so players can move immediately
		t.Type = core.TileGeneral // This tile becomes a general tile

		placements[pid] = placement
	}

	return placements, nil
}

func (g *Generator) findGeneralLocation(b *core.Board, existing []GeneralPlacement) (GeneralPlacement, error) {
	maxAttempts := b.W * b.H

	for range maxAttempts {
		x, y := g.rng.Intn(b.W), g.rng.Intn(b.H)
		idx := b.Idx(x, y)
		tile := &b.T[idx]

		// Generals must be placed on neutral, normal tiles.
		// Not on cities, existing generals, or mountains.
		if !tile.IsNeutral() || tile.Type != core.TileNormal {
			continue
		}

		// Check minimum distance from existing generals
		validLocation := true
		for _, other := range existing {
			otherX, otherY := b.XY(other.Idx)
			// Using Board's Distance method
			if b.Distance(x, y, otherX, otherY) < g.config.MinGeneralSpacing {
				validLocation = false
				break
			}
		}

		if validLocation {
			return GeneralPlacement{
				PlayerID: len(existing),
				Idx:      idx,
				X:        x,
				Y:        y,
			}, nil
		}
	}

	// Fallback: if truly no location respecting spacing is found after many attempts,
	// try to find *any* valid normal tile. This is less likely with proper spacing.
	for idx, tile := range b.T {
		if tile.IsNeutral() && tile.Type == core.TileNormal {
			// Check distance for fallback as well, if possible, but prioritize placement.
			// For simplicity in fallback, we might ignore spacing if initial attempts fail badly.
			// However, the problem description implies a robust system.
			// The provided fallback just picks the first available, which might violate spacing.
			// A better fallback would still try to respect spacing or have a relaxed spacing.
			// For now, keeping the original fallback logic structure:
			x, y := b.XY(idx)
			// Re-check spacing for fallback location if we want to be strict
			validFallbackLocation := true
			for _, other := range existing {
				otherX, otherY := b.XY(other.Idx)
				if b.Distance(x, y, otherX, otherY) < g.config.MinGeneralSpacing {
					validFallbackLocation = false
					break
				}
			}
			if validFallbackLocation {
				return GeneralPlacement{
					PlayerID: len(existing),
					Idx:      idx,
					X:        x,
					Y:        y,
				}, nil
			}
		}
	}
	// If even the fallback fails, it's a critical issue with map gen parameters or logic
	return GeneralPlacement{}, fmt.Errorf("unable to place general for player %d: no valid locations found on %dx%d map with %d existing generals (min spacing: %d)", len(existing), b.W, b.H, len(existing), g.config.MinGeneralSpacing)
}

// GeneralPlacement tracks where a general was placed
type GeneralPlacement struct {
	PlayerID int
	Idx      int
	X, Y     int
}
