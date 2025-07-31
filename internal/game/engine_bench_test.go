package game

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/rs/zerolog"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/mapgen"
)

func BenchmarkUpdatePlayerStats(b *testing.B) {
	testCases := []struct {
		name         string
		boardSize    int
		numPlayers   int
		changedTiles int
	}{
		{"Small_10x10_Few_Changes", 10, 2, 5},
		{"Small_10x10_Many_Changes", 10, 2, 50},
		{"Medium_20x20_Few_Changes", 20, 4, 10},
		{"Medium_20x20_Many_Changes", 20, 4, 200},
		{"Large_30x30_Few_Changes", 30, 4, 20},
		{"Large_30x30_Many_Changes", 30, 4, 450},
		{"XLarge_50x50_Few_Changes", 50, 8, 50},
		{"XLarge_50x50_Many_Changes", 50, 8, 1250},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			// Create engine with test board
			engine := createTestEngine(tc.boardSize, tc.numPlayers)
			
			// Set up some changed tiles
			setupChangedTiles(engine, tc.changedTiles)
			
			// Reset timer to exclude setup
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				// Clear changed tiles for next iteration
				engine.gs.ChangedTiles = make(map[int]struct{})
				setupChangedTiles(engine, tc.changedTiles)
				
				// Run the function we're benchmarking
				engine.updatePlayerStats()
			}
			
			b.ReportMetric(float64(tc.boardSize*tc.boardSize), "board_tiles")
			b.ReportMetric(float64(tc.changedTiles), "changed_tiles")
		})
	}
}

func createTestEngine(boardSize, numPlayers int) *Engine {
	// Create map generator
	mapConfig := mapgen.DefaultMapConfig(boardSize, boardSize, numPlayers)
	rng := rand.New(rand.NewSource(12345))
	gen := mapgen.NewGenerator(mapConfig, rng)
	
	// Generate board
	board, err := gen.GenerateMap()
	if err != nil {
		panic(fmt.Sprintf("Failed to generate map: %v", err))
	}
	
	// Find general positions
	playerStartPos := make([]struct{ X, Y int }, 0)
	for i, tile := range board.T {
		if tile.IsGeneral() {
			x, y := board.XY(i)
			playerStartPos = append(playerStartPos, struct{ X, Y int }{X: x, Y: y})
		}
	}
	
	players := make([]Player, numPlayers)
	for i := 0; i < numPlayers; i++ {
		players[i] = Player{
			ID:         i,
			Alive:      true,
			GeneralIdx: board.Idx(playerStartPos[i].X, playerStartPos[i].Y),
			OwnedTiles: make([]int, 0),
		}
	}
	
	gs := &GameState{
		Turn:                   1, // Skip turn 0 to avoid forced full update
		Board:                  board,
		Players:                players,
		ChangedTiles:           make(map[int]struct{}),
		VisibilityChangedTiles: make(map[int]struct{}),
	}
	
	// Create game config
	logger := zerolog.New(nil).Level(zerolog.Disabled)
	cfg := GameConfig{
		Width:   width,
		Height:  height,
		Players: numPlayers,
		Logger:  logger,
	}
	
	engine := NewEngine(context.Background(), cfg)
	if engine == nil {
		panic("Failed to create engine")
	}
	// Replace the game state with our test state
	engine.gs = gs
	// Initialize temp maps if not already done
	if engine.tempTileOwnership == nil {
		engine.tempTileOwnership = make(map[int]int)
	}
	if engine.tempAffectedPlayers == nil {
		engine.tempAffectedPlayers = make(map[int]struct{})
	}
	
	// Do initial full update
	engine.gs.Turn = 0
	engine.updatePlayerStats()
	engine.gs.Turn = 1
	
	return engine
}

func setupChangedTiles(engine *Engine, numChanges int) {
	boardSize := len(engine.gs.Board.T)
	if numChanges > boardSize {
		numChanges = boardSize
	}
	
	// Simulate some tile changes
	for i := 0; i < numChanges; i++ {
		tileIdx := i % boardSize
		// Simulate ownership change
		newOwner := i % len(engine.gs.Players)
		engine.gs.Board.T[tileIdx].Owner = newOwner
		engine.gs.Board.T[tileIdx].Army = i%50 + 1
		engine.gs.ChangedTiles[tileIdx] = struct{}{}
	}
}

func BenchmarkUpdatePlayerStats_WorstCase_FullScan(b *testing.B) {
	// Benchmark the old behavior by forcing full scan
	boardSizes := []int{10, 20, 30, 50}
	
	for _, size := range boardSizes {
		b.Run(fmt.Sprintf("Board_%dx%d", size, size), func(b *testing.B) {
			engine := createTestEngine(size, 4)
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				// Force full scan by setting turn to 0
				engine.gs.Turn = 0
				engine.updatePlayerStats()
				engine.gs.Turn = 1
			}
			
			b.ReportMetric(float64(size*size), "board_tiles")
		})
	}
}

func BenchmarkBoardStringBuilding(b *testing.B) {
	testCases := []struct {
		name      string
		boardSize int
		fogOfWar  bool
	}{
		{"Small_10x10_NoFog", 10, false},
		{"Small_10x10_WithFog", 10, true},
		{"Medium_20x20_NoFog", 20, false},
		{"Medium_20x20_WithFog", 20, true},
		{"Large_30x30_NoFog", 30, false},
		{"Large_30x30_WithFog", 30, true},
		{"XLarge_50x50_NoFog", 50, false},
		{"XLarge_50x50_WithFog", 50, true},
	}
	
	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			engine := createTestEngine(tc.boardSize, 4)
			engine.gs.FogOfWarEnabled = tc.fogOfWar
			
			// Set up some visibility if fog of war is enabled
			if tc.fogOfWar {
				// Make some tiles visible for player 0
				for i := 0; i < len(engine.gs.Board.T)/4; i++ {
					engine.gs.Board.T[i].SetVisible(0, true)
				}
			}
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_ = engine.Board(0) // Generate board string for player 0
			}
			
			b.ReportMetric(float64(tc.boardSize*tc.boardSize), "board_tiles")
		})
	}
}

func BenchmarkFogOfWarUpdate(b *testing.B) {
	testCases := []struct {
		name          string
		boardSize     int
		numPlayers    int
		changedTiles  int
	}{
		{"Small_10x10_Few_Changes", 10, 2, 2},
		{"Small_10x10_Many_Changes", 10, 2, 20},
		{"Medium_20x20_Few_Changes", 20, 4, 5},
		{"Medium_20x20_Many_Changes", 20, 4, 80},
		{"Large_30x30_Few_Changes", 30, 4, 10},
		{"Large_30x30_Many_Changes", 30, 4, 180},
		{"XLarge_50x50_Few_Changes", 50, 8, 20},
		{"XLarge_50x50_Many_Changes", 50, 8, 500},
	}
	
	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			engine := createTestEngine(tc.boardSize, tc.numPlayers)
			engine.gs.FogOfWarEnabled = true
			
			// Initialize with some owned tiles
			tilesPerPlayer := (tc.boardSize * tc.boardSize) / (tc.numPlayers * 2)
			for pid := 0; pid < tc.numPlayers; pid++ {
				for i := 0; i < tilesPerPlayer; i++ {
					tileIdx := (pid * tilesPerPlayer + i) % len(engine.gs.Board.T)
					engine.gs.Board.T[tileIdx].Owner = pid
				}
			}
			
			// Do initial full update
			engine.updatePlayerStats()
			engine.updateFogOfWar()
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				// Clear visibility changed tiles
				engine.gs.VisibilityChangedTiles = make(map[int]struct{})
				
				// Simulate some tiles changing ownership
				for j := 0; j < tc.changedTiles; j++ {
					tileIdx := (i*tc.changedTiles + j) % len(engine.gs.Board.T)
					engine.gs.VisibilityChangedTiles[tileIdx] = struct{}{}
				}
				
				engine.updateFogOfWar()
			}
			
			b.ReportMetric(float64(tc.boardSize*tc.boardSize), "board_tiles")
			b.ReportMetric(float64(tc.changedTiles), "changed_tiles")
		})
	}
}

func BenchmarkFogOfWar_FullUpdate(b *testing.B) {
	boardSizes := []int{10, 20, 30, 50}
	
	for _, size := range boardSizes {
		b.Run(fmt.Sprintf("Board_%dx%d", size, size), func(b *testing.B) {
			engine := createTestEngine(size, 4)
			engine.gs.FogOfWarEnabled = true
			
			// Set up owned tiles
			tilesPerPlayer := (size * size) / 8
			for pid := 0; pid < 4; pid++ {
				for i := 0; i < tilesPerPlayer; i++ {
					tileIdx := (pid * tilesPerPlayer + i) % len(engine.gs.Board.T)
					engine.gs.Board.T[tileIdx].Owner = pid
				}
			}
			
			engine.updatePlayerStats()
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				// Force full update
				engine.performFullVisibilityUpdate()
			}
			
			b.ReportMetric(float64(size*size), "board_tiles")
		})
	}
}

func BenchmarkMemoryAllocations(b *testing.B) {
	testCases := []struct {
		name         string
		boardSize    int
		numPlayers   int
		turnsToPlay  int
		tilesPerTurn int
	}{
		{"Small_Game_10_Turns", 20, 4, 10, 5},
		{"Medium_Game_50_Turns", 30, 4, 50, 10},
		{"Large_Game_100_Turns", 50, 8, 100, 20},
	}
	
	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				engine := createTestEngine(tc.boardSize, tc.numPlayers)
				engine.gs.FogOfWarEnabled = true
				
				// Simulate game turns
				for turn := 0; turn < tc.turnsToPlay; turn++ {
					// Clear tracking maps
					for k := range engine.gs.ChangedTiles {
						delete(engine.gs.ChangedTiles, k)
					}
					for k := range engine.gs.VisibilityChangedTiles {
						delete(engine.gs.VisibilityChangedTiles, k)
					}
					
					// Simulate tile changes
					for j := 0; j < tc.tilesPerTurn; j++ {
						tileIdx := (turn*tc.tilesPerTurn + j) % len(engine.gs.Board.T)
						newOwner := j % tc.numPlayers
						
						// Mark as changed
						engine.gs.ChangedTiles[tileIdx] = struct{}{}
						engine.gs.VisibilityChangedTiles[tileIdx] = struct{}{}
						
						// Change ownership
						engine.gs.Board.T[tileIdx].Owner = newOwner
					}
					
					// Update stats and visibility
					engine.updatePlayerStats()
					engine.updateFogOfWar()
				}
			}
			
			b.ReportMetric(float64(tc.boardSize*tc.boardSize), "board_tiles")
			b.ReportMetric(float64(tc.turnsToPlay), "turns_played")
		})
	}
}