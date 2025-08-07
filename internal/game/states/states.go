package states

import (
	"fmt"
	"time"
)

// InitializingState represents the game initialization phase
type InitializingState struct{}

func NewInitializingState() State {
	return &InitializingState{}
}

func (s *InitializingState) Phase() GamePhase {
	return PhaseInitializing
}

func (s *InitializingState) Enter(ctx *GameContext) error {
	ctx.Logger.Debug().Msg("Entering Initializing state")
	return nil
}

func (s *InitializingState) Exit(ctx *GameContext) error {
	ctx.Logger.Debug().Msg("Exiting Initializing state")
	return nil
}

func (s *InitializingState) Validate(ctx *GameContext) error {
	return nil
}

// LobbyState represents the lobby phase where players join
type LobbyState struct{}

func NewLobbyState() State {
	return &LobbyState{}
}

func (s *LobbyState) Phase() GamePhase {
	return PhaseLobby
}

func (s *LobbyState) Enter(ctx *GameContext) error {
	ctx.Logger.Info().Msg("Game lobby opened, waiting for players")
	return nil
}

func (s *LobbyState) Exit(ctx *GameContext) error {
	ctx.Logger.Info().
		Int("player_count", ctx.PlayerCount).
		Msg("Closing lobby, game starting")
	return nil
}

func (s *LobbyState) Validate(ctx *GameContext) error {
	if ctx.MaxPlayers < 1 {
		return fmt.Errorf("max players must be at least 1, got %d", ctx.MaxPlayers)
	}
	return nil
}

// StartingState represents the game setup phase
type StartingState struct{}

func NewStartingState() State {
	return &StartingState{}
}

func (s *StartingState) Phase() GamePhase {
	return PhaseStarting
}

func (s *StartingState) Enter(ctx *GameContext) error {
	ctx.Logger.Info().Msg("Starting game setup")
	// Map generation and player placement would happen here
	return nil
}

func (s *StartingState) Exit(ctx *GameContext) error {
	ctx.Logger.Debug().Msg("Game setup complete")
	return nil
}

func (s *StartingState) Validate(ctx *GameContext) error {
	if !ctx.IsReady() {
		return fmt.Errorf("not enough players to start: have %d, need at least 1", ctx.PlayerCount)
	}
	return nil
}

// RunningState represents active gameplay
type RunningState struct{}

func NewRunningState() State {
	return &RunningState{}
}

func (s *RunningState) Phase() GamePhase {
	return PhaseRunning
}

func (s *RunningState) Enter(ctx *GameContext) error {
	ctx.StartTime = time.Now()
	ctx.Logger.Info().
		Time("start_time", ctx.StartTime).
		Msg("Game started")
	return nil
}

func (s *RunningState) Exit(ctx *GameContext) error {
	elapsed := ctx.GetElapsedTime()
	ctx.Logger.Info().
		Dur("elapsed", elapsed).
		Msg("Exiting running state")
	return nil
}

func (s *RunningState) Validate(ctx *GameContext) error {
	if ctx.PlayerCount < 1 {
		return fmt.Errorf("cannot run game with no players")
	}
	return nil
}

// PausedState represents a paused game
type PausedState struct{}

func NewPausedState() State {
	return &PausedState{}
}

func (s *PausedState) Phase() GamePhase {
	return PhasePaused
}

func (s *PausedState) Enter(ctx *GameContext) error {
	ctx.PauseTime = time.Now()
	ctx.Logger.Info().
		Time("pause_time", ctx.PauseTime).
		Msg("Game paused")
	return nil
}

func (s *PausedState) Exit(ctx *GameContext) error {
	if !ctx.PauseTime.IsZero() {
		pauseDuration := time.Since(ctx.PauseTime)
		ctx.TotalPauseDuration += pauseDuration
		ctx.Logger.Info().
			Dur("pause_duration", pauseDuration).
			Dur("total_pause_duration", ctx.TotalPauseDuration).
			Msg("Game resumed")
	}
	return nil
}

func (s *PausedState) Validate(ctx *GameContext) error {
	if ctx.StartTime.IsZero() {
		return fmt.Errorf("cannot pause a game that hasn't started")
	}
	return nil
}

// EndingState represents the game ending phase
type EndingState struct{}

func NewEndingState() State {
	return &EndingState{}
}

func (s *EndingState) Phase() GamePhase {
	return PhaseEnding
}

func (s *EndingState) Enter(ctx *GameContext) error {
	ctx.Logger.Info().
		Int("winner", ctx.Winner).
		Msg("Game ending, determining final results")
	return nil
}

func (s *EndingState) Exit(ctx *GameContext) error {
	ctx.Logger.Debug().Msg("Game ending phase complete")
	return nil
}

func (s *EndingState) Validate(ctx *GameContext) error {
	if ctx.Winner < 0 && ctx.Error == nil {
		return fmt.Errorf("ending state requires either a winner or an error")
	}
	return nil
}

// EndedState represents a completed game
type EndedState struct{}

func NewEndedState() State {
	return &EndedState{}
}

func (s *EndedState) Phase() GamePhase {
	return PhaseEnded
}

func (s *EndedState) Enter(ctx *GameContext) error {
	elapsed := ctx.GetElapsedTime()
	ctx.Logger.Info().
		Int("winner", ctx.Winner).
		Dur("game_duration", elapsed).
		Msg("Game ended")
	return nil
}

func (s *EndedState) Exit(ctx *GameContext) error {
	ctx.Logger.Debug().Msg("Exiting ended state")
	return nil
}

func (s *EndedState) Validate(ctx *GameContext) error {
	return nil
}

// ErrorState represents an error condition
type ErrorState struct{}

func NewErrorState() State {
	return &ErrorState{}
}

func (s *ErrorState) Phase() GamePhase {
	return PhaseError
}

func (s *ErrorState) Enter(ctx *GameContext) error {
	ctx.Logger.Error().
		Err(ctx.Error).
		Msg("Game entered error state")
	return nil
}

func (s *ErrorState) Exit(ctx *GameContext) error {
	ctx.Logger.Info().Msg("Recovering from error state")
	ctx.Error = nil
	return nil
}

func (s *ErrorState) Validate(ctx *GameContext) error {
	if ctx.Error == nil {
		return fmt.Errorf("error state requires an error in context")
	}
	return nil
}

// ResetState represents game reset
type ResetState struct{}

func NewResetState() State {
	return &ResetState{}
}

func (s *ResetState) Phase() GamePhase {
	return PhaseReset
}

func (s *ResetState) Enter(ctx *GameContext) error {
	ctx.Logger.Info().Msg("Resetting game")

	// Clear game-specific data
	ctx.StartTime = time.Time{}
	ctx.PauseTime = time.Time{}
	ctx.TotalPauseDuration = 0
	ctx.Winner = -1
	ctx.Error = nil
	ctx.PlayerCount = 0

	// Clear metadata but keep the map allocated
	for k := range ctx.Metadata {
		delete(ctx.Metadata, k)
	}

	return nil
}

func (s *ResetState) Exit(ctx *GameContext) error {
	ctx.Logger.Debug().Msg("Game reset complete")
	return nil
}

func (s *ResetState) Validate(ctx *GameContext) error {
	return nil
}
