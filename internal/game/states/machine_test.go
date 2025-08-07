package states

import (
	"errors"
	"testing"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
)

func TestGamePhase_String(t *testing.T) {
	tests := []struct {
		phase    GamePhase
		expected string
	}{
		{PhaseInitializing, "Initializing"},
		{PhaseLobby, "Lobby"},
		{PhaseStarting, "Starting"},
		{PhaseRunning, "Running"},
		{PhasePaused, "Paused"},
		{PhaseEnding, "Ending"},
		{PhaseEnded, "Ended"},
		{PhaseError, "Error"},
		{PhaseReset, "Reset"},
		{GamePhase(999), "Unknown(999)"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.phase.String())
		})
	}
}

func TestGamePhase_Properties(t *testing.T) {
	t.Run("IsTerminal", func(t *testing.T) {
		assert.True(t, PhaseEnded.IsTerminal())
		assert.True(t, PhaseError.IsTerminal())
		assert.False(t, PhaseRunning.IsTerminal())
		assert.False(t, PhaseLobby.IsTerminal())
	})

	t.Run("CanReceiveActions", func(t *testing.T) {
		assert.True(t, PhaseRunning.CanReceiveActions())
		assert.False(t, PhaseLobby.CanReceiveActions())
		assert.False(t, PhasePaused.CanReceiveActions())
		assert.False(t, PhaseEnded.CanReceiveActions())
	})

	t.Run("CanAddPlayers", func(t *testing.T) {
		assert.True(t, PhaseLobby.CanAddPlayers())
		assert.False(t, PhaseRunning.CanAddPlayers())
		assert.False(t, PhaseInitializing.CanAddPlayers())
	})
}

func TestGamePhase_Transitions(t *testing.T) {
	tests := []struct {
		from    GamePhase
		allowed []GamePhase
	}{
		{PhaseInitializing, []GamePhase{PhaseLobby, PhaseError}},
		{PhaseLobby, []GamePhase{PhaseStarting, PhaseError}},
		{PhaseStarting, []GamePhase{PhaseRunning, PhaseError}},
		{PhaseRunning, []GamePhase{PhasePaused, PhaseEnding, PhaseError}},
		{PhasePaused, []GamePhase{PhaseRunning, PhaseEnding, PhaseError}},
		{PhaseEnding, []GamePhase{PhaseEnded, PhaseError}},
		{PhaseEnded, []GamePhase{PhaseReset}},
		{PhaseError, []GamePhase{PhaseReset}},
		{PhaseReset, []GamePhase{PhaseInitializing}},
	}

	for _, tt := range tests {
		t.Run(tt.from.String(), func(t *testing.T) {
			allowed := tt.from.AllowedTransitions()
			assert.Equal(t, tt.allowed, allowed)

			// Test CanTransitionTo
			for _, target := range tt.allowed {
				assert.True(t, tt.from.CanTransitionTo(target))
			}

			// Test invalid transitions
			allPhases := []GamePhase{
				PhaseInitializing, PhaseLobby, PhaseStarting, PhaseRunning,
				PhasePaused, PhaseEnding, PhaseEnded, PhaseError, PhaseReset,
			}
			for _, target := range allPhases {
				shouldAllow := false
				for _, allowed := range tt.allowed {
					if target == allowed {
						shouldAllow = true
						break
					}
				}
				assert.Equal(t, shouldAllow, tt.from.CanTransitionTo(target))
			}
		})
	}
}

func TestGameContext(t *testing.T) {
	logger := zerolog.New(zerolog.NewConsoleWriter()).Level(zerolog.DebugLevel)

	t.Run("NewGameContext", func(t *testing.T) {
		ctx := NewGameContext("test-game", 4, logger)
		assert.Equal(t, "test-game", ctx.GameID)
		assert.Equal(t, 4, ctx.MaxPlayers)
		assert.Equal(t, -1, ctx.Winner)
		assert.NotNil(t, ctx.Metadata)
	})

	t.Run("IsReady", func(t *testing.T) {
		ctx := NewGameContext("test-game", 4, logger)

		ctx.PlayerCount = 0
		assert.False(t, ctx.IsReady())

		ctx.PlayerCount = 1
		assert.True(t, ctx.IsReady())

		ctx.PlayerCount = 2
		assert.True(t, ctx.IsReady())

		ctx.PlayerCount = 4
		assert.True(t, ctx.IsReady())

		ctx.PlayerCount = 5
		assert.False(t, ctx.IsReady())
	})

	t.Run("GetElapsedTime", func(t *testing.T) {
		ctx := NewGameContext("test-game", 4, logger)

		// No start time
		assert.Equal(t, time.Duration(0), ctx.GetElapsedTime())

		// With start time
		ctx.StartTime = time.Now().Add(-10 * time.Second)
		elapsed := ctx.GetElapsedTime()
		assert.Greater(t, elapsed, 9*time.Second)
		assert.Less(t, elapsed, 11*time.Second)

		// With pause duration
		ctx.TotalPauseDuration = 5 * time.Second
		elapsed = ctx.GetElapsedTime()
		assert.Greater(t, elapsed, 4*time.Second)
		assert.Less(t, elapsed, 6*time.Second)
	})

	t.Run("Metadata", func(t *testing.T) {
		ctx := NewGameContext("test-game", 4, logger)

		// Set and get metadata
		ctx.SetMetadata("key1", "value1")
		ctx.SetMetadata("key2", 42)

		val1, exists1 := ctx.GetMetadata("key1")
		assert.True(t, exists1)
		assert.Equal(t, "value1", val1)

		val2, exists2 := ctx.GetMetadata("key2")
		assert.True(t, exists2)
		assert.Equal(t, 42, val2)

		_, exists3 := ctx.GetMetadata("nonexistent")
		assert.False(t, exists3)
	})
}

func TestStateMachine(t *testing.T) {
	logger := zerolog.New(zerolog.NewConsoleWriter()).Level(zerolog.DebugLevel)

	setup := func() (*StateMachine, *GameContext) {
		ctx := NewGameContext("test-game", 4, logger)
		eventBus := events.NewEventBus()
		sm := NewStateMachine(ctx, eventBus)
		return sm, ctx
	}

	t.Run("NewStateMachine", func(t *testing.T) {
		sm, _ := setup()
		assert.Equal(t, PhaseInitializing, sm.CurrentPhase())
		assert.NotNil(t, sm.states)
		assert.Len(t, sm.states, 9) // All default states registered
	})

	t.Run("Valid Transitions", func(t *testing.T) {
		sm, ctx := setup()

		// Initializing -> Lobby
		err := sm.TransitionTo(PhaseLobby, "test transition")
		assert.NoError(t, err)
		assert.Equal(t, PhaseLobby, sm.CurrentPhase())

		// Lobby -> Starting (need players)
		ctx.PlayerCount = 2
		err = sm.TransitionTo(PhaseStarting, "players ready")
		assert.NoError(t, err)
		assert.Equal(t, PhaseStarting, sm.CurrentPhase())

		// Starting -> Running
		err = sm.TransitionTo(PhaseRunning, "setup complete")
		assert.NoError(t, err)
		assert.Equal(t, PhaseRunning, sm.CurrentPhase())
		assert.False(t, ctx.StartTime.IsZero())

		// Running -> Paused
		err = sm.TransitionTo(PhasePaused, "user paused")
		assert.NoError(t, err)
		assert.Equal(t, PhasePaused, sm.CurrentPhase())
		assert.False(t, ctx.PauseTime.IsZero())

		// Paused -> Running
		time.Sleep(10 * time.Millisecond) // Ensure some pause duration
		err = sm.TransitionTo(PhaseRunning, "user resumed")
		assert.NoError(t, err)
		assert.Equal(t, PhaseRunning, sm.CurrentPhase())
		assert.Greater(t, ctx.TotalPauseDuration, time.Duration(0))

		// Running -> Ending
		ctx.Winner = 1
		err = sm.TransitionTo(PhaseEnding, "player 1 won")
		assert.NoError(t, err)
		assert.Equal(t, PhaseEnding, sm.CurrentPhase())

		// Ending -> Ended
		err = sm.TransitionTo(PhaseEnded, "game over")
		assert.NoError(t, err)
		assert.Equal(t, PhaseEnded, sm.CurrentPhase())
	})

	t.Run("Invalid Transitions", func(t *testing.T) {
		sm, _ := setup()

		// Cannot go directly from Initializing to Running
		err := sm.TransitionTo(PhaseRunning, "skip steps")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid transition")
		assert.Equal(t, PhaseInitializing, sm.CurrentPhase())

		// Cannot go from Lobby to Running
		_ = sm.TransitionTo(PhaseLobby, "setup")
		err = sm.TransitionTo(PhaseRunning, "skip starting")
		assert.Error(t, err)
		assert.Equal(t, PhaseLobby, sm.CurrentPhase())
	})

	t.Run("State Validation", func(t *testing.T) {
		sm, ctx := setup()

		// Cannot start without any players
		_ = sm.TransitionTo(PhaseLobby, "setup")
		ctx.PlayerCount = 0
		ctx.MaxPlayers = 0 // Also set MaxPlayers to 0 to trigger validation
		err := sm.TransitionTo(PhaseStarting, "no players")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not enough players")

		// Cannot pause before starting
		ctx.PlayerCount = 2
		ctx.MaxPlayers = 4 // Restore MaxPlayers
		_ = sm.TransitionTo(PhaseStarting, "enough players")
		ctx.StartTime = time.Time{} // Clear start time
		err = sm.TransitionTo(PhasePaused, "pause without start")
		assert.Error(t, err)

		// Cannot enter error state without error
		_ = sm.TransitionTo(PhaseRunning, "start game")
		err = sm.TransitionTo(PhaseError, "no error")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "requires an error")
	})

	t.Run("History Tracking", func(t *testing.T) {
		sm, ctx := setup()

		// Make several transitions
		_ = sm.TransitionTo(PhaseLobby, "reason1")
		ctx.PlayerCount = 2
		_ = sm.TransitionTo(PhaseStarting, "reason2")
		_ = sm.TransitionTo(PhaseRunning, "reason3")

		history := sm.GetHistory()
		assert.Len(t, history, 3)

		assert.Equal(t, PhaseInitializing, history[0].From)
		assert.Equal(t, PhaseLobby, history[0].To)
		assert.Equal(t, "reason1", history[0].Reason)

		assert.Equal(t, PhaseLobby, history[1].From)
		assert.Equal(t, PhaseStarting, history[1].To)
		assert.Equal(t, "reason2", history[1].Reason)

		assert.Equal(t, PhaseStarting, history[2].From)
		assert.Equal(t, PhaseRunning, history[2].To)
		assert.Equal(t, "reason3", history[2].Reason)
	})

	t.Run("Error State Recovery", func(t *testing.T) {
		sm, ctx := setup()

		// Enter error state
		sm.TransitionTo(PhaseLobby, "setup")
		ctx.Error = errors.New("test error")
		err := sm.TransitionTo(PhaseError, "error occurred")
		assert.NoError(t, err)
		assert.Equal(t, PhaseError, sm.CurrentPhase())

		// Reset from error
		err = sm.TransitionTo(PhaseReset, "recover from error")
		assert.NoError(t, err)
		assert.Equal(t, PhaseReset, sm.CurrentPhase())
		assert.Nil(t, ctx.Error) // Error cleared on exit

		// Back to initializing
		err = sm.TransitionTo(PhaseInitializing, "restart")
		assert.NoError(t, err)
		assert.Equal(t, PhaseInitializing, sm.CurrentPhase())
	})

	t.Run("CanTransitionTo", func(t *testing.T) {
		sm, _ := setup()

		assert.True(t, sm.CanTransitionTo(PhaseLobby))
		assert.True(t, sm.CanTransitionTo(PhaseError))
		assert.False(t, sm.CanTransitionTo(PhaseRunning))
		assert.False(t, sm.CanTransitionTo(PhaseEnded))
	})
}

// MockState for testing custom state implementations
type MockState struct {
	phase       GamePhase
	enterCalled bool
	exitCalled  bool
	enterError  error
	exitError   error
}

func (m *MockState) Phase() GamePhase            { return m.phase }
func (m *MockState) Enter(*GameContext) error    { m.enterCalled = true; return m.enterError }
func (m *MockState) Exit(*GameContext) error     { m.exitCalled = true; return m.exitError }
func (m *MockState) Validate(*GameContext) error { return nil }

func TestStateMachine_CustomStates(t *testing.T) {
	logger := zerolog.New(zerolog.NewConsoleWriter()).Level(zerolog.DebugLevel)
	ctx := NewGameContext("test-game", 4, logger)
	eventBus := events.NewEventBus()
	sm := NewStateMachine(ctx, eventBus)

	t.Run("RegisterCustomState", func(t *testing.T) {
		mockState := &MockState{phase: GamePhase(100)}
		sm.RegisterState(mockState)

		// Verify state is registered
		assert.Contains(t, sm.states, GamePhase(100))
	})

	t.Run("StateCallbacks", func(t *testing.T) {
		// Create mock states for lobby and starting
		lobbyMock := &MockState{phase: PhaseLobby}
		startingMock := &MockState{phase: PhaseStarting}

		sm.RegisterState(lobbyMock)
		sm.RegisterState(startingMock)

		// Transition to lobby
		err := sm.TransitionTo(PhaseLobby, "test")
		assert.NoError(t, err)
		assert.True(t, lobbyMock.enterCalled)
		assert.False(t, lobbyMock.exitCalled)

		// Transition to starting
		ctx.PlayerCount = 2
		err = sm.TransitionTo(PhaseStarting, "test")
		assert.NoError(t, err)
		assert.True(t, lobbyMock.exitCalled)
		assert.True(t, startingMock.enterCalled)
	})
}
