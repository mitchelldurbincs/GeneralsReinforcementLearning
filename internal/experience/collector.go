package experience

import (
	"sync"

	"github.com/google/uuid"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// SimpleCollector implements a basic in-memory experience collector
type SimpleCollector struct {
	experiences []*experiencepb.Experience
	mu          sync.Mutex
	maxSize     int
	gameID      string
	serializer  *Serializer
	logger      zerolog.Logger
}

// NewSimpleCollector creates a new simple experience collector
func NewSimpleCollector(maxSize int, gameID string, logger zerolog.Logger) *SimpleCollector {
	return &SimpleCollector{
		experiences: make([]*experiencepb.Experience, 0, maxSize),
		maxSize:     maxSize,
		gameID:      gameID,
		serializer:  NewSerializer(),
		logger:      logger.With().Str("component", "experience_collector").Logger(),
	}
}

// OnStateTransition collects experience from a state transition
func (c *SimpleCollector) OnStateTransition(prevState, currState *game.GameState, actions map[int]*game.Action) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Skip if buffer is full
	if len(c.experiences) >= c.maxSize {
		c.logger.Warn().
			Int("buffer_size", len(c.experiences)).
			Int("max_size", c.maxSize).
			Msg("Experience buffer full, dropping experience")
		return
	}

	// Process experience for each player that took an action
	for playerID, action := range actions {
		if action == nil {
			continue
		}

		// Generate unique experience ID
		expID := uuid.New().String()

		// Serialize states from player's perspective
		stateTensor := c.serializer.StateToTensor(prevState, playerID)
		nextStateTensor := c.serializer.StateToTensor(currState, playerID)

		// Calculate reward
		reward := CalculateReward(prevState, currState, playerID)

		// Generate action mask for legal moves
		actionMask := c.serializer.GenerateActionMask(prevState, playerID)

		// Convert action to flattened index
		actionIndex := c.serializer.ActionToIndex(action, prevState.Board.W)

		// Check if game ended
		done := currState.IsGameOver()

		// Create experience protobuf
		exp := &experiencepb.Experience{
			ExperienceId: expID,
			GameId:       c.gameID,
			PlayerId:     int32(playerID),
			Turn:         int32(currState.Turn),
			State: &experiencepb.TensorState{
				Shape: []int32{NumChannels, int32(prevState.Board.H), int32(prevState.Board.W)},
				Data:  stateTensor,
			},
			Action: int32(actionIndex),
			Reward: reward,
			NextState: &experiencepb.TensorState{
				Shape: []int32{NumChannels, int32(currState.Board.H), int32(currState.Board.W)},
				Data:  nextStateTensor,
			},
			Done:        done,
			ActionMask:  actionMask,
			CollectedAt: timestamppb.Now(),
			Metadata: map[string]string{
				"collector_version": "1.0.0",
			},
		}

		c.experiences = append(c.experiences, exp)

		c.logger.Debug().
			Str("experience_id", expID).
			Int("player_id", playerID).
			Int("turn", currState.Turn).
			Float32("reward", reward).
			Bool("done", done).
			Msg("Collected experience")
	}
}

// OnGameEnd handles terminal states
func (c *SimpleCollector) OnGameEnd(finalState *game.GameState) {
	c.logger.Info().
		Str("game_id", c.gameID).
		Int("total_experiences", len(c.experiences)).
		Int("winner", finalState.GetWinner()).
		Int("final_turn", finalState.Turn).
		Msg("Game ended, finalizing experience collection")

	// Could add terminal experience processing here if needed
}

// GetExperiences returns a copy of all collected experiences
func (c *SimpleCollector) GetExperiences() []*experiencepb.Experience {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Return a copy to avoid external modifications
	result := make([]*experiencepb.Experience, len(c.experiences))
	copy(result, c.experiences)
	return result
}

// GetExperienceCount returns the current number of experiences
func (c *SimpleCollector) GetExperienceCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.experiences)
}

// Clear removes all experiences from the buffer
func (c *SimpleCollector) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.experiences = c.experiences[:0]
}

// GetLatestExperiences returns the n most recent experiences
func (c *SimpleCollector) GetLatestExperiences(n int) []*experiencepb.Experience {
	c.mu.Lock()
	defer c.mu.Unlock()

	if n > len(c.experiences) {
		n = len(c.experiences)
	}

	start := len(c.experiences) - n
	result := make([]*experiencepb.Experience, n)
	copy(result, c.experiences[start:])
	return result
}