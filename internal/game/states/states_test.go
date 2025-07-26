package states

import (
	"errors"
	"testing"
	"time"
	
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
)

func TestStateImplementations(t *testing.T) {
	logger := zerolog.New(zerolog.NewConsoleWriter()).Level(zerolog.DebugLevel)
	
	t.Run("InitializingState", func(t *testing.T) {
		state := NewInitializingState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhaseInitializing, state.Phase())
		assert.NoError(t, state.Enter(ctx))
		assert.NoError(t, state.Exit(ctx))
		assert.NoError(t, state.Validate(ctx))
	})
	
	t.Run("LobbyState", func(t *testing.T) {
		state := NewLobbyState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhaseLobby, state.Phase())
		assert.NoError(t, state.Enter(ctx))
		assert.NoError(t, state.Exit(ctx))
		assert.NoError(t, state.Validate(ctx))
		
		// Test validation failure
		ctx.MaxPlayers = 1
		err := state.Validate(ctx)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "max players must be at least 2")
	})
	
	t.Run("StartingState", func(t *testing.T) {
		state := NewStartingState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhaseStarting, state.Phase())
		
		// Test validation failure - not enough players
		ctx.PlayerCount = 1
		err := state.Validate(ctx)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "not enough players")
		
		// Test success
		ctx.PlayerCount = 2
		assert.NoError(t, state.Validate(ctx))
		assert.NoError(t, state.Enter(ctx))
		assert.NoError(t, state.Exit(ctx))
	})
	
	t.Run("RunningState", func(t *testing.T) {
		state := NewRunningState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhaseRunning, state.Phase())
		
		// Test validation
		ctx.PlayerCount = 1
		err := state.Validate(ctx)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot run game with fewer than 2 players")
		
		ctx.PlayerCount = 2
		assert.NoError(t, state.Validate(ctx))
		
		// Test enter sets start time
		assert.True(t, ctx.StartTime.IsZero())
		assert.NoError(t, state.Enter(ctx))
		assert.False(t, ctx.StartTime.IsZero())
		
		assert.NoError(t, state.Exit(ctx))
	})
	
	t.Run("PausedState", func(t *testing.T) {
		state := NewPausedState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhasePaused, state.Phase())
		
		// Test validation - cannot pause before starting
		err := state.Validate(ctx)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "cannot pause a game that hasn't started")
		
		ctx.StartTime = time.Now()
		assert.NoError(t, state.Validate(ctx))
		
		// Test enter sets pause time
		assert.True(t, ctx.PauseTime.IsZero())
		assert.NoError(t, state.Enter(ctx))
		assert.False(t, ctx.PauseTime.IsZero())
		
		// Test exit updates pause duration
		time.Sleep(10 * time.Millisecond)
		assert.Equal(t, time.Duration(0), ctx.TotalPauseDuration)
		assert.NoError(t, state.Exit(ctx))
		assert.Greater(t, ctx.TotalPauseDuration, time.Duration(0))
	})
	
	t.Run("EndingState", func(t *testing.T) {
		state := NewEndingState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhaseEnding, state.Phase())
		
		// Test validation - needs winner or error
		err := state.Validate(ctx)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "requires either a winner or an error")
		
		// With winner
		ctx.Winner = 1
		assert.NoError(t, state.Validate(ctx))
		assert.NoError(t, state.Enter(ctx))
		assert.NoError(t, state.Exit(ctx))
		
		// With error
		ctx.Winner = -1
		ctx.Error = errors.New("test error")
		assert.NoError(t, state.Validate(ctx))
	})
	
	t.Run("EndedState", func(t *testing.T) {
		state := NewEndedState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhaseEnded, state.Phase())
		assert.NoError(t, state.Validate(ctx))
		
		// Test with winner and elapsed time
		ctx.StartTime = time.Now().Add(-1 * time.Hour)
		ctx.Winner = 2
		assert.NoError(t, state.Enter(ctx))
		assert.NoError(t, state.Exit(ctx))
	})
	
	t.Run("ErrorState", func(t *testing.T) {
		state := NewErrorState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhaseError, state.Phase())
		
		// Test validation - needs error
		err := state.Validate(ctx)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "error state requires an error")
		
		ctx.Error = errors.New("test error")
		assert.NoError(t, state.Validate(ctx))
		assert.NoError(t, state.Enter(ctx))
		
		// Exit should clear error
		assert.NotNil(t, ctx.Error)
		assert.NoError(t, state.Exit(ctx))
		assert.Nil(t, ctx.Error)
	})
	
	t.Run("ResetState", func(t *testing.T) {
		state := NewResetState()
		ctx := NewGameContext("test", 4, logger)
		
		assert.Equal(t, PhaseReset, state.Phase())
		assert.NoError(t, state.Validate(ctx))
		
		// Set up some state
		ctx.StartTime = time.Now()
		ctx.PauseTime = time.Now()
		ctx.TotalPauseDuration = 5 * time.Second
		ctx.Winner = 3
		ctx.Error = errors.New("test")
		ctx.PlayerCount = 4
		ctx.SetMetadata("key", "value")
		
		// Enter should reset everything
		assert.NoError(t, state.Enter(ctx))
		
		assert.True(t, ctx.StartTime.IsZero())
		assert.True(t, ctx.PauseTime.IsZero())
		assert.Equal(t, time.Duration(0), ctx.TotalPauseDuration)
		assert.Equal(t, -1, ctx.Winner)
		assert.Nil(t, ctx.Error)
		assert.Equal(t, 0, ctx.PlayerCount)
		
		_, exists := ctx.GetMetadata("key")
		assert.False(t, exists)
		
		assert.NoError(t, state.Exit(ctx))
	})
}