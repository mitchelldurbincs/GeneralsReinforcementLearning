package gameserver

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/states"
	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

func TestGameCreationWithStateMachine(t *testing.T) {
	// Create game manager
	gm := NewGameManager(10)
	
	// Create game config
	config := &gamev1.GameConfig{
		Width:      10,
		Height:     10,
		MaxPlayers: 2,
		FogOfWar:   true,
	}
	
	// Create game
	game, gameID, err := gm.CreateGame(config)
	require.NoError(t, err)
	require.NotNil(t, game)
	require.NotEmpty(t, gameID)
	
	// Verify state machine was initialized
	assert.NotNil(t, game.stateMachine, "State machine should be initialized")
	assert.NotNil(t, game.eventBus, "Event bus should be initialized")
	
	// Verify initial state is Lobby
	currentPhase := game.stateMachine.CurrentPhase()
	assert.Equal(t, states.PhaseLobby, currentPhase, "Game should be in Lobby phase after creation")
	
	// Verify phase can be retrieved through CurrentPhase method
	protoPhase := game.CurrentPhase()
	assert.Equal(t, commonv1.GamePhase_GAME_PHASE_LOBBY, protoPhase, "Proto phase should be LOBBY")
}

func TestStateMachinePhaseValidation(t *testing.T) {
	// Create game manager
	gm := NewGameManager(10)
	
	// Create game
	config := &gamev1.GameConfig{
		Width:      10,
		Height:     10,
		MaxPlayers: 2,
	}
	
	game, _, err := gm.CreateGame(config)
	require.NoError(t, err)
	
	// Verify Lobby phase allows adding players
	assert.True(t, game.stateMachine.CurrentPhase().CanAddPlayers(), "Lobby phase should allow adding players")
	
	// Verify Lobby phase doesn't allow receiving actions
	assert.False(t, game.stateMachine.CurrentPhase().CanReceiveActions(), "Lobby phase should not allow receiving actions")
}

func TestEventBusSubscription(t *testing.T) {
	// Create game manager
	gm := NewGameManager(10)
	
	// Create game
	config := &gamev1.GameConfig{
		Width:      10,
		Height:     10,
		MaxPlayers: 2,
	}
	
	game, _, err := gm.CreateGame(config)
	require.NoError(t, err)
	
	// Track state transitions
	transitionCount := 0
	var lastTransition *events.StateTransitionEvent
	
	// Subscribe to state transition events
	game.eventBus.SubscribeFunc("state.transition", func(event events.Event) {
		if e, ok := event.(*events.StateTransitionEvent); ok {
			transitionCount++
			lastTransition = e
		}
	})
	
	// Add a player to satisfy the state machine validation
	playerID, _, added := game.AddPlayer("TestPlayer")
	require.True(t, added, "Should be able to add player")
	require.Equal(t, int32(0), playerID, "First player should have ID 0")
	
	// Now trigger a state transition (this would normally happen when all players join)
	err = game.stateMachine.TransitionTo(states.PhaseStarting, "Test transition")
	require.NoError(t, err)
	
	// Verify event was published (we should have received at least one transition)
	assert.Greater(t, transitionCount, 0, "Should have received state transition events")
	assert.NotNil(t, lastTransition, "Should have captured transition event")
	if lastTransition != nil {
		assert.Equal(t, "Starting", lastTransition.ToPhase, "Should have transitioned to Starting phase")
	}
}