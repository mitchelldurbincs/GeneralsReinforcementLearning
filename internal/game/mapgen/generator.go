package mapgen

import (
	"math/rand"
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
}

// DefaultMapConfig returns a sensible default configuration
func DefaultMapConfig(w, h, players int) MapConfig {
	return MapConfig{
		Width:             w,
		Height:            h,
		PlayerCount:       players,
		CityRatio:         20,
		CityStartArmy:     40,
		MinGeneralSpacing: 5,
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

// GenerateMap creates a new board with cities and generals placed
func (g *Generator) GenerateMap() *core.Board {
	board := core.NewBoard(g.config.Width, g.config.Height)
	
	g.placeCities(board)
	g.placeGenerals(board)
	
	return board
}

func (g *Generator) placeCities(b *core.Board) {
	want := (b.W * b.H) / g.config.CityRatio
	placed := 0
	
	// Use a maximum attempt counter to avoid infinite loops
	maxAttempts := want * 10
	attempts := 0
	
	for placed < want && attempts < maxAttempts {
		x, y := g.rng.Intn(b.W), g.rng.Intn(b.H)
		idx := b.Idx(x, y)
		t := &b.T[idx]
		
		if t.IsNeutral() && t.Type == core.TileNormal {
			t.Type = core.TileCity
			t.Army = g.config.CityStartArmy
			placed++
		}
		attempts++
	}
}

func (g *Generator) placeGenerals(b *core.Board) []GeneralPlacement {
	placements := make([]GeneralPlacement, g.config.PlayerCount)
	
	for pid := 0; pid < g.config.PlayerCount; pid++ {
		placement := g.findGeneralLocation(b, placements[:pid])
		
		t := &b.T[placement.Idx]
		t.Owner = pid
		t.Army = 1
		t.Type = core.TileGeneral
		
		placements[pid] = placement
	}
	
	return placements
}

func (g *Generator) findGeneralLocation(b *core.Board, existing []GeneralPlacement) GeneralPlacement {
	maxAttempts := b.W * b.H // Fallback to prevent infinite loops
	
	for attempts := 0; attempts < maxAttempts; attempts++ {
		x, y := g.rng.Intn(b.W), g.rng.Intn(b.H)
		idx := b.Idx(x, y)
		
		if !b.T[idx].IsNeutral() || b.T[idx].IsCity() {
			continue
		}
		
		// Check minimum distance from existing generals
		validLocation := true
		for _, other := range existing {
			otherX, otherY := b.XY(other.Idx)
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
			}
		}
	}
	
	// Fallback: place anywhere valid (shouldn't happen with reasonable configs)
	for idx, tile := range b.T {
		if tile.IsNeutral() && tile.Type == core.TileNormal {
			x, y := b.XY(idx)
			return GeneralPlacement{
				PlayerID: len(existing),
				Idx:      idx,
				X:        x,
				Y:        y,
			}
		}
	}
	
	panic("Unable to place general - no valid locations")
}

// GeneralPlacement tracks where a general was placed
type GeneralPlacement struct {
	PlayerID int
	Idx      int
	X, Y     int
}