package states

import (
	"fmt"
	"sync"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
)

// State represents a game state with lifecycle callbacks
type State interface {
	// Phase returns the GamePhase this state represents
	Phase() GamePhase

	// Enter is called when transitioning into this state
	Enter(ctx *GameContext) error

	// Exit is called when transitioning out of this state
	Exit(ctx *GameContext) error

	// Validate checks if the state is valid given the context
	Validate(ctx *GameContext) error
}

// Transition represents a state transition in the history
type Transition struct {
	From      GamePhase
	To        GamePhase
	Timestamp time.Time
	Reason    string
}

// StateMachine manages game state transitions and history
type StateMachine struct {
	mu             sync.RWMutex
	currentPhase   GamePhase
	states         map[GamePhase]State
	context        *GameContext
	history        []Transition
	maxHistorySize int
	eventBus       *events.EventBus
}

// NewStateMachine creates a new state machine
func NewStateMachine(ctx *GameContext, eventBus *events.EventBus) *StateMachine {
	sm := &StateMachine{
		currentPhase:   PhaseInitializing,
		states:         make(map[GamePhase]State),
		context:        ctx,
		history:        make([]Transition, 0, 100),
		maxHistorySize: 1000,
		eventBus:       eventBus,
	}

	// Register default states
	sm.registerDefaultStates()

	return sm
}

// registerDefaultStates registers the built-in state implementations
func (sm *StateMachine) registerDefaultStates() {
	// We'll implement these state structs next
	sm.RegisterState(NewInitializingState())
	sm.RegisterState(NewLobbyState())
	sm.RegisterState(NewStartingState())
	sm.RegisterState(NewRunningState())
	sm.RegisterState(NewPausedState())
	sm.RegisterState(NewEndingState())
	sm.RegisterState(NewEndedState())
	sm.RegisterState(NewErrorState())
	sm.RegisterState(NewResetState())
}

// RegisterState registers a state implementation
func (sm *StateMachine) RegisterState(state State) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.states[state.Phase()] = state
}

// CurrentPhase returns the current game phase
func (sm *StateMachine) CurrentPhase() GamePhase {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return sm.currentPhase
}

// TransitionTo attempts to transition to the specified phase
func (sm *StateMachine) TransitionTo(targetPhase GamePhase, reason string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Check if transition is allowed
	if !sm.currentPhase.CanTransitionTo(targetPhase) {
		return fmt.Errorf("invalid transition from %s to %s", sm.currentPhase, targetPhase)
	}

	// Get state implementations
	currentState, hasCurrentState := sm.states[sm.currentPhase]
	targetState, hasTargetState := sm.states[targetPhase]

	if !hasTargetState {
		return fmt.Errorf("no state implementation for phase %s", targetPhase)
	}

	// Validate target state
	if err := targetState.Validate(sm.context); err != nil {
		return fmt.Errorf("target state validation failed: %w", err)
	}

	// Exit current state
	if hasCurrentState {
		if err := currentState.Exit(sm.context); err != nil {
			sm.context.Logger.Error().
				Err(err).
				Str("from_phase", sm.currentPhase.String()).
				Str("to_phase", targetPhase.String()).
				Msg("Error exiting state")
			// Continue with transition despite exit error
		}
	}

	// Record transition
	transition := Transition{
		From:      sm.currentPhase,
		To:        targetPhase,
		Timestamp: time.Now(),
		Reason:    reason,
	}
	sm.addToHistory(transition)

	// Update current phase
	previousPhase := sm.currentPhase
	sm.currentPhase = targetPhase

	// Enter new state
	if err := targetState.Enter(sm.context); err != nil {
		// Rollback on enter failure
		sm.currentPhase = previousPhase
		return fmt.Errorf("failed to enter state %s: %w", targetPhase, err)
	}

	// Publish state transition event
	if sm.eventBus != nil {
		sm.eventBus.Publish(events.NewStateTransitionEvent(
			sm.context.GameID,
			previousPhase.String(),
			targetPhase.String(),
			reason,
		))
	}

	sm.context.Logger.Info().
		Str("from_phase", previousPhase.String()).
		Str("to_phase", targetPhase.String()).
		Str("reason", reason).
		Msg("State transition completed")

	return nil
}

// addToHistory adds a transition to the history, maintaining max size
func (sm *StateMachine) addToHistory(transition Transition) {
	sm.history = append(sm.history, transition)

	// Trim history if it exceeds max size
	if len(sm.history) > sm.maxHistorySize {
		// Keep the most recent entries
		sm.history = sm.history[len(sm.history)-sm.maxHistorySize:]
	}
}

// GetHistory returns a copy of the transition history
func (sm *StateMachine) GetHistory() []Transition {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	history := make([]Transition, len(sm.history))
	copy(history, sm.history)
	return history
}

// GetContext returns the game context
func (sm *StateMachine) GetContext() *GameContext {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return sm.context
}

// CanTransitionTo checks if a transition to the target phase is allowed
func (sm *StateMachine) CanTransitionTo(targetPhase GamePhase) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return sm.currentPhase.CanTransitionTo(targetPhase)
}

// Reset clears the history and resets to initial state
func (sm *StateMachine) Reset() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.history = sm.history[:0]
	return sm.TransitionTo(PhaseInitializing, "Reset requested")
}
