package game

import (
	"math/rand"
	"testing"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Define GameState and Player structs here if they are not exported
// or to ensure test isolation, though they are accessible via e.gs.
// For this test, we'll assume they are defined as in your engine.go
// (i.e., accessible as e.gs.Players, e.gs.Board).

// Helper to create a deterministic RNG for tests
func newTestRNG() *rand.Rand {
	return rand.New(rand.NewSource(12345)) // Fixed seed for deterministic tests
}

func TestNewEngine(t *testing.T) {
	rng := newTestRNG()
	width, height, numPlayers := 8, 8, 2

	engine := NewEngine(width, height, numPlayers, rng)

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
	engine := NewEngine(5, 5, 1, newTestRNG())
	initialTurn := engine.gs.Turn
	initialArmy := engine.gs.Players[0].ArmyCount

	err := engine.Step(nil) // No actions
	require.NoError(t, err)

	assert.Equal(t, initialTurn+1, engine.gs.Turn, "Turn should increment")
	// Assuming GeneralProduction = 1 from your engine's internal constants
	assert.Equal(t, initialArmy+1, engine.gs.Players[0].ArmyCount, "Player army should increase due to general production")
	assert.False(t, engine.IsGameOver(), "Game should not be over with 1 player and no actions")
}

func TestEngine_Step_GameOverReturnError(t *testing.T) {
	engine := NewEngine(5, 5, 1, newTestRNG())
	engine.gameOver = true // Manually set game to over

	err := engine.Step(nil)
	assert.ErrorIs(t, err, core.ErrGameOver, "Step should return ErrGameOver if game is already over")
}

func TestEngine_ProcessTurnProduction(t *testing.T) {
	// To test this effectively, we need to know/control the production constants.
	// Assuming: GeneralProduction=1, CityProduction=1, NormalProduction=1, NormalGrowInterval=3 (for test)
	// Since we can't easily override package-level unexported consts, this test is illustrative.
	// You'd adapt it based on your actual production values.

	// For this test, we'll assume the production constants used in `engine.go` are:
	// GeneralProduction = 1
	// CityProduction = 1
	// NormalProduction = 1
	// NormalGrowInterval = 25 (This is from your previous engine.go thoughts)
	// If these are different, the expected values below must change.

	engine := NewEngine(5, 5, 1, newTestRNG())
	playerID := 0
	p := &engine.gs.Players[playerID]

	// Find player's general
	require.NotEqual(t, -1, p.GeneralIdx, "Player should have a general")
	generalTile := &engine.gs.Board.T[p.GeneralIdx]
	initialGeneralArmy := generalTile.Army

	// Create a city for the player
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

	// Create a normal land tile for the player
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
	
	// Simulate a turn that triggers normal production
	// We need to know the actual NormalGrowInterval from your engine.go
	// Let's assume it's 25, as discussed.
	const actualNormalGrowInterval = 25 // Replace with your engine's actual constant
	const actualGeneralProduction = 1
	const actualCityProduction = 1
	const actualNormalProduction = 1


	engine.gs.Turn = actualNormalGrowInterval -1 // So next turn (Turn % Interval == 0)
	engine.updatePlayerStats() // Update stats before production for accurate initial counts

	engine.processTurnProduction() // Call directly for focused test

	assert.Equal(t, initialGeneralArmy+actualGeneralProduction, generalTile.Army, "General production mismatch")
	assert.Equal(t, initialCityArmy+actualCityProduction, cityTile.Army, "City production mismatch")
	if engine.gs.Turn % actualNormalGrowInterval == 0 { // This will be true given Turn setup
		assert.Equal(t, initialLandArmy+actualNormalProduction, landTile.Army, "Normal land production mismatch")
	} else {
		assert.Equal(t, initialLandArmy, landTile.Army, "Normal land should not produce on this turn")
	}
}

func TestEngine_PlayerEliminationAndTileTurnover(t *testing.T) {
	engine := NewEngine(5, 5, 2, newTestRNG()) // Player 0 and Player 1
	p0, p1 := &engine.gs.Players[0], &engine.gs.Players[1]

	// Manually set up a scenario for general capture
	// P0: (0,0) Army 20, Type Normal
	// P1: General at (0,1) Army 1, Type General
	// P1: City at (1,1) Army 5, Type City, Owner P1
	// P1: Land at (2,2) Army 3, Type Normal, Owner P1

	board := engine.gs.Board
	p0AttackerIdx := board.Idx(0,0)
	board.T[p0AttackerIdx] = core.Tile{Owner: 0, Army: 20, Type: core.TileNormal}

	// Ensure P1's general is placed by mapgen, find it, then move it for test control if necessary
	// Or, more simply, directly overwrite a tile to be P1's general for this test.
	// For this test, we'll assume mapgen placed generals somewhere. We'll find P1's general.
	// A more robust test might directly place generals to specific locations.
	
	var p1GeneralOriginalIdx = p1.GeneralIdx
	require.NotEqual(t, -1, p1GeneralOriginalIdx, "Player 1 should have a general from NewEngine")

	// We will make tile (0,1) P1's general for this specific test setup for predictability
	p1NewGeneralIdx := board.Idx(0,1)
	if p1GeneralOriginalIdx != p1NewGeneralIdx { // If mapgen put it elsewhere
		board.T[p1GeneralOriginalIdx].Owner = core.NeutralID // Neutralize original spot
		board.T[p1GeneralOriginalIdx].Type = core.TileNormal
		board.T[p1GeneralOriginalIdx].Army = 0
	}
	board.T[p1NewGeneralIdx] = core.Tile{Owner: 1, Army: 1, Type: core.TileGeneral}
	p1.GeneralIdx = p1NewGeneralIdx // Update player struct to reflect this controlled position

	p1CityIdx := board.Idx(1,1)
	board.T[p1CityIdx] = core.Tile{Owner: 1, Army: 5, Type: core.TileCity}
	p1LandIdx := board.Idx(2,2)
	board.T[p1LandIdx] = core.Tile{Owner: 1, Army: 3, Type: core.TileNormal}

	// Ensure P0 is not on top of P1's general if mapgen was very unlucky
	if p0.GeneralIdx == p1NewGeneralIdx {
		t.Fatal("Test setup error: P0 general is on P1 general spot")
	}
	if p0AttackerIdx == p1NewGeneralIdx {
		t.Fatal("Test setup error: P0 attacker is on P1 general spot")
	}
	
	engine.updatePlayerStats() // Recalculate based on our manual setup

	// Action: P0 from (0,0) captures P1's General at (0,1)
	action := &core.MoveAction{
		PlayerID: 0, FromX: 0, FromY: 0, ToX: 0, ToY: 1, MoveAll: true,
	}

	// Simulate a Step
	err := engine.Step([]core.Action{action})
	require.NoError(t, err, "Step should not error during capture")

	// --- Assertions ---
	// Player 1 should be eliminated
	assert.False(t, engine.gs.Players[1].Alive, "Player 1 should be eliminated (Alive=false)")
	assert.Equal(t, -1, engine.gs.Players[1].GeneralIdx, "Player 1 should have no general index")

	// Player 0 should be alive
	assert.True(t, engine.gs.Players[0].Alive, "Player 0 should still be alive")

	// Check tile ownership turnover
	// P1's original general tile (0,1) should now be P0's
	assert.Equal(t, 0, board.T[p1NewGeneralIdx].Owner, "Captured general tile (0,1) should be owned by Player 0")
	// Armies: 20 (P0) - 1 (left behind) - 1 (P1 general) = 18
	assert.Equal(t, 18, board.T[p1NewGeneralIdx].Army, "Army count on captured general tile mismatch")

	// P1's other tiles should now be P0's
	assert.Equal(t, 0, board.T[p1CityIdx].Owner, "Player 1's city at (1,1) should now be owned by Player 0")
	assert.Equal(t, 5, board.T[p1CityIdx].Army, "Player 1's city at (1,1) army count should remain") // Armies remain
	assert.Equal(t, 0, board.T[p1LandIdx].Owner, "Player 1's land at (2,2) should now be owned by Player 0")
	assert.Equal(t, 3, board.T[p1LandIdx].Army, "Player 1's land at (2,2) army count should remain")

	// Game should be over, P0 wins
	assert.True(t, engine.IsGameOver(), "Game should be over after elimination")
	assert.Equal(t, 0, engine.GetWinner(), "Player 0 should be the winner")
}

func TestEngine_Step_ActionFromDeadPlayer(t *testing.T) {
	engine := NewEngine(5, 5, 2, newTestRNG())
	player0ID := 0
	player1ID := 1

	// Manually mark player 1 as dead
	engine.gs.Players[player1ID].Alive = false
	engine.gs.Players[player1ID].GeneralIdx = -1 // Ensure no general for dead player

	// P0 is alive and owns (0,0) with 10 army
	board := engine.gs.Board
	p0TileIdx := board.Idx(0,0)
	board.T[p0TileIdx].Owner = player0ID
	board.T[p0TileIdx].Army = 10
	
	// P1 attempts an action from a tile they 'own' but they are dead
	// (even if P1 owns no tiles, the check is on Alive status first)
	p1OwnedTileIdx := board.Idx(1,1) // Let's say P1 still formally owns a tile
	board.T[p1OwnedTileIdx].Owner = player1ID
	board.T[p1OwnedTileIdx].Army = 5


	actions := []core.Action{
		&core.MoveAction{PlayerID: player1ID, FromX: 1, FromY: 1, ToX: 1, ToY: 2, MoveAll: true}, // Action from dead P1
		&core.MoveAction{PlayerID: player0ID, FromX: 0, FromY: 0, ToX: 0, ToY: 1, MoveAll: true}, // Valid action from P0
	}
	
	initialP0ArmyOnTile := board.T[p0TileIdx].Army
	initialP1ArmyOnTile := board.T[p1OwnedTileIdx].Army

	err := engine.Step(actions)
	require.NoError(t, err)

	// P1's action (dead player) should have been ignored
	assert.Equal(t, player1ID, board.T[p1OwnedTileIdx].Owner, "P1's tile ownership should not change from their own invalid action")
	assert.Equal(t, initialP1ArmyOnTile, board.T[p1OwnedTileIdx].Army, "P1's tile army should not change from their own invalid action")


	// P0's action should have processed (assuming (0,1) is a valid target)
	// P0 moves from (0,0) to (0,1)
	assert.Equal(t, 1, board.T[p0TileIdx].Army, "P0's original tile should have 1 army left")
	
	p0TargetIdx := board.Idx(0,1)
	// Assuming (0,1) was neutral with 0 army
	assert.Equal(t, player0ID, board.T[p0TargetIdx].Owner, "P0 should own target tile (0,1)")
	assert.Equal(t, initialP0ArmyOnTile -1, board.T[p0TargetIdx].Army, "P0's target tile (0,1) army count is wrong")


	assert.True(t, engine.gs.Players[player0ID].Alive, "Player 0 should be alive")
	assert.False(t, engine.gs.Players[player1ID].Alive, "Player 1 should remain dead")
}

// Add more tests:
// - Test GetWinner when no one has won yet (game not over)
// - Test GetWinner in a draw (if possible by rules, current rules make it unlikely with <=1 alive)
// - Test Step with an action that fails validation (e.g., move to mountain) - ensure game state is consistent
// - Test edge cases for production (e.g., turn 0, turn NormalGrowInterval-1, turn NormalGrowInterval)