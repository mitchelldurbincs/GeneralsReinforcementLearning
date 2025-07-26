# Game State Machine

This package implements a finite state machine for managing game lifecycle and transitions in the Generals Reinforcement Learning project.

## Overview

The state machine ensures proper game flow and prevents invalid operations by enforcing state transitions and validations. It provides:

- Clear game phase definitions
- Enforced transition rules
- State lifecycle callbacks (Enter/Exit)
- Transition history tracking
- Event publication for state changes
- Thread-safe operations

## Game Phases

```
PhaseInitializing → PhaseLobby → PhaseStarting → PhaseRunning ⇄ PhasePaused
                        ↓             ↓               ↓              ↓
                    PhaseError    PhaseError     PhaseEnding → PhaseEnded
                        ↓                             ↓              ↓
                    PhaseReset ←──────────────────────┴──────────────┘
                        ↓
                PhaseInitializing
```

### Phase Descriptions

- **PhaseInitializing**: Game object creation and initial setup
- **PhaseLobby**: Players joining, waiting for game to start (can add players)
- **PhaseStarting**: Map generation, player placement
- **PhaseRunning**: Active gameplay (can receive player actions)
- **PhasePaused**: Temporary game suspension
- **PhaseEnding**: Winner determination, cleanup in progress
- **PhaseEnded**: Final state, game complete
- **PhaseError**: Error recovery state
- **PhaseReset**: Reset game without full teardown

## Usage

### Basic Setup

```go
import (
    "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/states"
    "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
)

// Create game context
ctx := states.NewGameContext("game-123", maxPlayers, logger)

// Create event bus for state transition events
eventBus := events.NewEventBus()

// Create state machine
sm := states.NewStateMachine(ctx, eventBus)
```

### Transitions

```go
// Move to lobby
err := sm.TransitionTo(states.PhaseLobby, "Game initialized")

// Add players
ctx.PlayerCount = 2

// Start game when ready
if ctx.IsReady() {
    err := sm.TransitionTo(states.PhaseStarting, "All players ready")
}

// Begin gameplay
err := sm.TransitionTo(states.PhaseRunning, "Setup complete")

// Pause game
err := sm.TransitionTo(states.PhasePaused, "Player requested pause")

// Resume game
err := sm.TransitionTo(states.PhaseRunning, "Player resumed")

// End game
ctx.Winner = playerID
err := sm.TransitionTo(states.PhaseEnding, "Player won")
err := sm.TransitionTo(states.PhaseEnded, "Game complete")
```

### Error Handling

```go
// Enter error state
ctx.Error = fmt.Errorf("critical game error")
err := sm.TransitionTo(states.PhaseError, "Error occurred")

// Recover via reset
err := sm.TransitionTo(states.PhaseReset, "Resetting from error")
err := sm.TransitionTo(states.PhaseInitializing, "Starting fresh")
```

### Checking State

```go
// Get current phase
phase := sm.CurrentPhase()

// Check capabilities
if phase.CanReceiveActions() {
    // Process player action
}

if phase.CanAddPlayers() {
    // Allow player to join
}

// Check if transition is allowed
if sm.CanTransitionTo(states.PhaseRunning) {
    // Transition is valid
}
```

### History

```go
// Get transition history
history := sm.GetHistory()
for _, transition := range history {
    log.Printf("Transitioned from %s to %s at %v: %s",
        transition.From, transition.To, 
        transition.Timestamp, transition.Reason)
}
```

## Integration with Game Engine

To integrate the state machine with the game engine:

1. Add state machine to Engine struct:
```go
type Engine struct {
    // ... existing fields
    stateMachine *states.StateMachine
}
```

2. Initialize in NewEngine:
```go
ctx := states.NewGameContext(gameID, maxPlayers, logger)
e.stateMachine = states.NewStateMachine(ctx, e.eventBus)
```

3. Guard operations by state:
```go
func (e *Engine) ProcessAction(action Action) error {
    if !e.stateMachine.CurrentPhase().CanReceiveActions() {
        return fmt.Errorf("cannot process actions in %s phase", 
            e.stateMachine.CurrentPhase())
    }
    // ... process action
}
```

4. Transition on game events:
```go
// When game starts
err := e.stateMachine.TransitionTo(states.PhaseRunning, "Game started")

// When winner determined
e.stateMachine.GetContext().Winner = winnerID
err := e.stateMachine.TransitionTo(states.PhaseEnding, "Winner determined")
```

## Events

State transitions publish `StateTransitionEvent` to the event bus:

```go
type StateTransitionEvent struct {
    FromPhase string
    ToPhase   string
    Reason    string
}
```

Subscribe to track state changes:
```go
eventBus.Subscribe(func(event events.Event) {
    if e, ok := event.(*events.StateTransitionEvent); ok {
        log.Printf("State changed: %s → %s (%s)", 
            e.FromPhase, e.ToPhase, e.Reason)
    }
})
```

## Thread Safety

The state machine is thread-safe. All public methods use appropriate locking to ensure concurrent access is safe.

## Testing

See `machine_test.go` and `states_test.go` for comprehensive examples of:
- State transitions
- Validation logic
- Error handling
- Custom state implementations
- History tracking