package game

// ExperienceCollector is an interface for collecting experiences during gameplay
type ExperienceCollector interface {
	// OnStateTransition is called after each game state transition
	OnStateTransition(prevState, currState *GameState, actions map[int]*Action)

	// OnGameEnd is called when the game ends
	OnGameEnd(finalState *GameState)
}
