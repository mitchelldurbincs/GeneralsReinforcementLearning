package states

import "fmt"

// GamePhase represents the current phase of a game
type GamePhase int

const (
	// PhaseInitializing - Game object creation
	PhaseInitializing GamePhase = iota

	// PhaseLobby - Players joining, configuration
	PhaseLobby

	// PhaseStarting - Map generation, player placement
	PhaseStarting

	// PhaseRunning - Active gameplay
	PhaseRunning

	// PhasePaused - Temporary suspension
	PhasePaused

	// PhaseEnding - Winner determination, cleanup
	PhaseEnding

	// PhaseEnded - Final state
	PhaseEnded

	// PhaseError - Error recovery state
	PhaseError

	// PhaseReset - Reset the current game without full teardown
	PhaseReset
)

// String returns the string representation of a GamePhase
func (p GamePhase) String() string {
	switch p {
	case PhaseInitializing:
		return "Initializing"
	case PhaseLobby:
		return "Lobby"
	case PhaseStarting:
		return "Starting"
	case PhaseRunning:
		return "Running"
	case PhasePaused:
		return "Paused"
	case PhaseEnding:
		return "Ending"
	case PhaseEnded:
		return "Ended"
	case PhaseError:
		return "Error"
	case PhaseReset:
		return "Reset"
	default:
		return fmt.Sprintf("Unknown(%d)", p)
	}
}

// IsTerminal returns true if the phase represents a terminal state
func (p GamePhase) IsTerminal() bool {
	return p == PhaseEnded || p == PhaseError
}

// CanReceiveActions returns true if the game can process player actions in this phase
func (p GamePhase) CanReceiveActions() bool {
	return p == PhaseRunning
}

// CanAddPlayers returns true if players can join in this phase
func (p GamePhase) CanAddPlayers() bool {
	return p == PhaseLobby
}

// AllowedTransitions returns the valid phases this phase can transition to
func (p GamePhase) AllowedTransitions() []GamePhase {
	switch p {
	case PhaseInitializing:
		return []GamePhase{PhaseLobby, PhaseError}
	case PhaseLobby:
		return []GamePhase{PhaseStarting, PhaseError}
	case PhaseStarting:
		return []GamePhase{PhaseRunning, PhaseError}
	case PhaseRunning:
		return []GamePhase{PhasePaused, PhaseEnding, PhaseError}
	case PhasePaused:
		return []GamePhase{PhaseRunning, PhaseEnding, PhaseError}
	case PhaseEnding:
		return []GamePhase{PhaseEnded, PhaseError}
	case PhaseEnded:
		return []GamePhase{PhaseReset}
	case PhaseError:
		return []GamePhase{PhaseReset}
	case PhaseReset:
		return []GamePhase{PhaseInitializing}
	default:
		return []GamePhase{}
	}
}

// CanTransitionTo checks if a transition from this phase to the target phase is allowed
func (p GamePhase) CanTransitionTo(target GamePhase) bool {
	allowed := p.AllowedTransitions()
	for _, phase := range allowed {
		if phase == target {
			return true
		}
	}
	return false
}

// ParsePhase converts a string to a GamePhase
func ParsePhase(s string) GamePhase {
	switch s {
	case "Initializing":
		return PhaseInitializing
	case "Lobby":
		return PhaseLobby
	case "Starting":
		return PhaseStarting
	case "Running":
		return PhaseRunning
	case "Paused":
		return PhasePaused
	case "Ending":
		return PhaseEnding
	case "Ended":
		return PhaseEnded
	case "Error":
		return PhaseError
	case "Reset":
		return PhaseReset
	default:
		return PhaseInitializing // Default to initializing for unknown phases
	}
}
