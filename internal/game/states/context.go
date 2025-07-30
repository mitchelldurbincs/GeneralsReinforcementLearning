package states

import (
	"time"
	
	"github.com/rs/zerolog"
)

// GameContext provides game-specific information to states for making decisions
type GameContext struct {
	// GameID uniquely identifies this game instance
	GameID string
	
	// Logger for state-specific logging
	Logger zerolog.Logger
	
	// PlayerCount is the number of players in the game
	PlayerCount int
	
	// MaxPlayers is the maximum number of players allowed
	MaxPlayers int
	
	// StartTime is when the game started (PhaseRunning entered)
	StartTime time.Time
	
	// PauseTime is when the game was paused (if paused)
	PauseTime time.Time
	
	// TotalPauseDuration tracks total time spent paused
	TotalPauseDuration time.Duration
	
	// Winner is the player ID of the winner (if game ended)
	Winner int
	
	// Error holds any error that caused transition to PhaseError
	Error error
	
	// Metadata for custom state data
	Metadata map[string]interface{}
}

// NewGameContext creates a new game context
func NewGameContext(gameID string, maxPlayers int, logger zerolog.Logger) *GameContext {
	return &GameContext{
		GameID:     gameID,
		MaxPlayers: maxPlayers,
		Logger:     logger.With().Str("game_id", gameID).Logger(),
		Metadata:   make(map[string]interface{}),
		Winner:     -1, // -1 indicates no winner yet
	}
}

// IsReady returns true if the game has enough players to start
func (gc *GameContext) IsReady() bool {
	return gc.PlayerCount >= 1 && gc.PlayerCount <= gc.MaxPlayers
}

// GetElapsedTime returns the time elapsed since game start, excluding pauses
func (gc *GameContext) GetElapsedTime() time.Duration {
	if gc.StartTime.IsZero() {
		return 0
	}
	
	elapsed := time.Since(gc.StartTime)
	return elapsed - gc.TotalPauseDuration
}

// SetMetadata stores custom data for states
func (gc *GameContext) SetMetadata(key string, value interface{}) {
	gc.Metadata[key] = value
}

// GetMetadata retrieves custom data stored by states
func (gc *GameContext) GetMetadata(key string) (interface{}, bool) {
	val, exists := gc.Metadata[key]
	return val, exists
}