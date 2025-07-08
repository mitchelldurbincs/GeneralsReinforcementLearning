package game

import (
	"context" // Added for context.Context
	"math/rand"
	"testing"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestRNG creates a deterministic RNG for tests.
func newTestRNG() *rand.Rand {
	return rand.New(rand.NewSource(12345)) // Fixed seed
}

// testLogger provides a Nop logger for tests where log output is not asserted.
func testLogger() zerolog.Logger {
	return zerolog.Nop()
}

func TestNewEngine(t *testing.T) {
	rng := newTestRNG()
	width, height, numPlayers := 8, 8, 2
	ctx := context.Background()

	config := GameConfig{
		Width:   width,
		Height:  height,
		Players: numPlayers,
		Rng:     rng,
		Logger:  testLogger(),
	}
	engine := NewEngine(ctx, config)

	require.NotNil(t, engine, "Engine should not be nil")
	require.NotNil(t, engine.gs, "GameState should not be nil")
	require.NotNil(t, engine.gs.Board, "Board should not be nil")
	assert.Equal(t, width, engine.gs.Board.W, "Board width mismatch")
	assert.Equal(t, height, engine.gs.Board.H, "Board height mismatch")

	require.Len(t, engine.gs.Players, numPlayers, "Incorrect number of players")
	assert.False(t, engine.gameOver, "Game should not be over at start")
	assert.Equal(t, 0, engine.gs.Turn, "Initial turn should be 0")

	generalsFound := 0
	for i := 0; i < numPlayers; i++ {
		player := engine.gs.Players[i]
		assert.True(t, player.Alive, "Player %d should be alive", i)
		assert.NotEqual(t, -1, player.GeneralIdx, "Player %d should have a general assigned", i)
		if player.GeneralIdx != -1 {
			generalTile := engine.gs.Board.T[player.GeneralIdx]
			assert.Equal(t, i, generalTile.Owner, "General tile owner mismatch for player %d", i)
			assert.True(t, generalTile.IsGeneral(), "General tile type mismatch for player %d", i)
			generalsFound++
		}
		assert.True(t, player.ArmyCount >= 1, "Player %d should have at least 1 army (their general)", i)
	}
	assert.Equal(t, numPlayers, generalsFound, "All players should have a general on the board")
}

func TestEngine_Step_BasicTurn(t *testing.T) {
	ctx := context.Background()
	config := GameConfig{
		Width:   5,
		Height:  5,
		Players: 1,
		Rng:     newTestRNG(),
		Logger:  testLogger(),
	}
	engine := NewEngine(ctx, config)
	initialTurn := engine.gs.Turn
	initialArmy := engine.gs.Players[0].ArmyCount

	err := engine.Step(ctx, nil) // No actions
	require.NoError(t, err)

	assert.Equal(t, initialTurn+1, engine.gs.Turn, "Turn should increment")
	// Assuming GeneralProduction = 1 (defined in engine.go or core)
	assert.Equal(t, initialArmy+GeneralProduction, engine.gs.Players[0].ArmyCount, "Player army should increase due to general production")
	assert.False(t, engine.IsGameOver(), "Game should not be over with 1 player and no actions")
}

func TestEngine_Step_GameOverReturnError(t *testing.T) {
	ctx := context.Background()
	config := GameConfig{
		Width:   5,
		Height:  5,
		Players: 1,
		Rng:     newTestRNG(),
		Logger:  testLogger(),
	}
	engine := NewEngine(ctx, config)
	engine.gameOver = true // Manually set game to over

	err := engine.Step(ctx, nil)
	assert.ErrorIs(t, err, core.ErrGameOver, "Step should return ErrGameOver if game is already over")
}

func TestEngine_ProcessTurnProduction(t *testing.T) {
	ctx := context.Background()
	// Production constants should ideally be exported from your game package or core
	// if they need to be used precisely in tests. For now, using test-local constants.
	const (
		actualNormalGrowInterval = 25
		actualGeneralProduction  = 1
		actualCityProduction     = 1
		actualNormalProduction   = 1
	)

	config := GameConfig{
		Width:   5,
		Height:  5,
		Players: 1,
		Rng:     newTestRNG(),
		Logger:  testLogger(),
	}
	engine := NewEngine(ctx, config)
	playerID := 0
	p := &engine.gs.Players[playerID]

	require.NotEqual(t, -1, p.GeneralIdx, "Player should have a general")
	generalTile := &engine.gs.Board.T[p.GeneralIdx]
	initialGeneralArmy := generalTile.Army

	cityIdx := -1
	for i, tile := range engine.gs.Board.T {
		if tile.Type == core.TileNormal && tile.IsNeutral() {
			engine.gs.Board.T[i].Owner = playerID
			engine.gs.Board.T[i].Type = core.TileCity
			engine.gs.Board.T[i].Army = 5
			cityIdx = i
			break
		}
	}
	require.NotEqual(t, -1, cityIdx, "Could not place a test city")
	cityTile := &engine.gs.Board.T[cityIdx]
	initialCityArmy := cityTile.Army

	landIdx := -1
	for i, tile := range engine.gs.Board.T {
		if tile.Type == core.TileNormal && tile.IsNeutral() && i != cityIdx && i != p.GeneralIdx {
			engine.gs.Board.T[i].Owner = playerID
			engine.gs.Board.T[i].Army = 2
			landIdx = i
			break
		}
	}
	require.NotEqual(t, -1, landIdx, "Could not place test land")
	landTile := &engine.gs.Board.T[landIdx]
	initialLandArmy := landTile.Army

	engine.updatePlayerStats() // Update stats before production for accurate initial counts

	// Scenario 1: Normal production *should* occur
	engine.gs.Turn = actualNormalGrowInterval // e.g., Turn 25 if interval is 25
	currentGeneralArmy := generalTile.Army
	currentCityArmy := cityTile.Army
	currentLandArmy := landTile.Army
	engine.processTurnProduction(testLogger()) // processTurnProduction takes the engine's logger, or a passed one
	assert.Equal(t, currentGeneralArmy+actualGeneralProduction, generalTile.Army, "General production mismatch (prod turn)")
	assert.Equal(t, currentCityArmy+actualCityProduction, cityTile.Army, "City production mismatch (prod turn)")
	assert.Equal(t, currentLandArmy+actualNormalProduction, landTile.Army, "Normal land production mismatch when it should produce")

	// Scenario 2: Normal production *should NOT* occur
	engine.gs.Turn = actualNormalGrowInterval - 1 // e.g., Turn 24 if interval is 25
	generalTile.Army = initialGeneralArmy         // Reset armies
	cityTile.Army = initialCityArmy
	landTile.Army = initialLandArmy
	currentGeneralArmy = generalTile.Army
	currentCityArmy = cityTile.Army
	currentLandArmy = landTile.Army

	engine.processTurnProduction(testLogger())
	assert.Equal(t, currentGeneralArmy+actualGeneralProduction, generalTile.Army, "General production mismatch (no-normal-prod turn)")
	assert.Equal(t, currentCityArmy+actualCityProduction, cityTile.Army, "City production mismatch (no-normal-prod turn)")
	assert.Equal(t, currentLandArmy, landTile.Army, "Normal land should not produce on this turn")
}

func TestEngine_PlayerEliminationAndTileTurnover(t *testing.T) {
	ctx := context.Background()
	config := GameConfig{
		Width:   5,
		Height:  5,
		Players: 2,
		Rng:     newTestRNG(),
		Logger:  testLogger(),
	}
	engine := NewEngine(ctx, config) // Player 0 and Player 1
	p0, p1 := &engine.gs.Players[0], &engine.gs.Players[1]

	// Setup: P0 attacks P1's General
	// P0: (0,0) Army 20, Type Normal
	// P1: General at (0,1) Army 1
	// P1: City at (1,1) Army 5, Owner P1
	// P1: Land at (2,2) Army 3, Owner P1
	board := engine.gs.Board
	p0AttackerIdx := board.Idx(0, 0)
	board.T[p0AttackerIdx] = core.Tile{Owner: 0, Army: 20, Type: core.TileNormal, Visible: make(map[int]bool)}

	p1GeneralOriginalIdx := p1.GeneralIdx
	require.NotEqual(t, -1, p1GeneralOriginalIdx, "Player 1 should have a general from NewEngine")

	p1NewGeneralIdx := board.Idx(0, 1)
	if p1GeneralOriginalIdx != p1NewGeneralIdx { // Relocate P1's general for test predictability
		board.T[p1GeneralOriginalIdx] = core.Tile{Owner: core.NeutralID, Type: core.TileNormal, Army: 0, Visible: make(map[int]bool)}
	}
	board.T[p1NewGeneralIdx] = core.Tile{Owner: 1, Army: 1, Type: core.TileGeneral, Visible: make(map[int]bool)}
	p1.GeneralIdx = p1NewGeneralIdx

	p1CityIdx := board.Idx(1, 1)
	board.T[p1CityIdx] = core.Tile{Owner: 1, Army: 5, Type: core.TileCity, Visible: make(map[int]bool)}
	p1LandIdx := board.Idx(2, 2)
	board.T[p1LandIdx] = core.Tile{Owner: 1, Army: 3, Type: core.TileNormal, Visible: make(map[int]bool)}

	require.NotEqual(t, p0.GeneralIdx, p1NewGeneralIdx, "Test setup conflict: P0 general on P1 general spot")
	require.NotEqual(t, p0AttackerIdx, p1NewGeneralIdx, "Test setup conflict: P0 attacker on P1 general spot")

	engine.updatePlayerStats()

	action := &core.MoveAction{
		PlayerID: 0, FromX: 0, FromY: 0, ToX: 0, ToY: 1, MoveAll: true,
	}

	err := engine.Step(ctx, []core.Action{action})
	require.NoError(t, err, "Step should not error during capture")

	assert.False(t, engine.gs.Players[1].Alive, "Player 1 should be eliminated")
	assert.Equal(t, -1, engine.gs.Players[1].GeneralIdx, "Player 1 should have no general index")
	assert.True(t, engine.gs.Players[0].Alive, "Player 0 should still be alive")

	// Assert tile ownership and army counts post-capture
	// Note: Step() applies production after actions, so generals and cities get +1
	assert.Equal(t, 0, board.T[p1NewGeneralIdx].Owner, "Captured general tile should be owned by Player 0")
	assert.Equal(t, 19, board.T[p1NewGeneralIdx].Army, "Army count on captured general tile (20 - 1 left - 1 defender + 1 production)")

	assert.Equal(t, 0, board.T[p1CityIdx].Owner, "Player 1's city should now be owned by Player 0")
	assert.Equal(t, 6, board.T[p1CityIdx].Army, "Player 1's city army count (5 + 1 production)")
	assert.Equal(t, 0, board.T[p1LandIdx].Owner, "Player 1's land should now be owned by Player 0")
	assert.Equal(t, 3, board.T[p1LandIdx].Army, "Player 1's land army count should remain (no production for normal tiles on turn 1)")

	assert.True(t, engine.IsGameOver(), "Game should be over after elimination")
	assert.Equal(t, 0, engine.GetWinner(), "Player 0 should be the winner")
}

func TestEngine_Step_ActionFromDeadPlayer(t *testing.T) {
	ctx := context.Background()
	config := GameConfig{
		Width:   5,
		Height:  5,
		Players: 2,
		Rng:     newTestRNG(),
		Logger:  testLogger(),
	}
	engine := NewEngine(ctx, config)
	player0ID := 0
	player1ID := 1

	// First, remove P1's general from the board to truly make them dead
	if engine.gs.Players[player1ID].GeneralIdx != -1 {
		board := engine.gs.Board
		board.T[engine.gs.Players[player1ID].GeneralIdx] = core.Tile{Owner: core.NeutralID, Army: 0, Type: core.TileNormal, Visible: make(map[int]bool)}
	}
	
	engine.gs.Players[player1ID].Alive = false
	engine.gs.Players[player1ID].GeneralIdx = -1

	board := engine.gs.Board
	p0TileIdx := board.Idx(0, 0)
	board.T[p0TileIdx] = core.Tile{Owner: player0ID, Army: 10, Type: core.TileNormal, Visible: make(map[int]bool)}

	p1OwnedTileIdx := board.Idx(1, 1) // P1 formally owns this tile despite being dead
	board.T[p1OwnedTileIdx] = core.Tile{Owner: player1ID, Army: 5, Type: core.TileNormal, Visible: make(map[int]bool)}

	actions := []core.Action{
		&core.MoveAction{PlayerID: player1ID, FromX: 1, FromY: 1, ToX: 1, ToY: 2, MoveAll: true}, // Action from dead P1
		&core.MoveAction{PlayerID: player0ID, FromX: 0, FromY: 0, ToX: 0, ToY: 1, MoveAll: true}, // Valid action from P0
	}

	initialP0ArmyOnTile := board.T[p0TileIdx].Army
	initialP1ArmyOnTile := board.T[p1OwnedTileIdx].Army

	err := engine.Step(ctx, actions)
	require.NoError(t, err)

	// Assert P1's action was ignored
	assert.Equal(t, player1ID, board.T[p1OwnedTileIdx].Owner, "P1's tile ownership should not change from their own invalid action")
	assert.Equal(t, initialP1ArmyOnTile, board.T[p1OwnedTileIdx].Army, "P1's tile army should not change from their own invalid action")

	// Assert P0's action processed (assuming (0,1) is neutral and empty for simplicity)
	assert.Equal(t, 1, board.T[p0TileIdx].Army, "P0's original tile should have 1 army left after moving all")
	p0TargetIdx := board.Idx(0, 1)
	targetTile := &board.T[p0TargetIdx]
	assert.Equal(t, player0ID, targetTile.Owner, "P0 should own target tile (0,1)")
	assert.Equal(t, initialP0ArmyOnTile-1, targetTile.Army, "P0's target tile (0,1) army count is wrong")

	assert.True(t, engine.gs.Players[player0ID].Alive, "Player 0 should be alive")
	assert.False(t, engine.gs.Players[player1ID].Alive, "Player 1 should remain dead")
}

