package experience

import (
	"github.com/google/uuid"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// SimpleCollector implements a basic in-memory experience collector using a ring buffer
type SimpleCollector struct {
	buffer     *Buffer
	gameID     string
	serializer *Serializer
	logger     zerolog.Logger
}

// NewSimpleCollector creates a new simple experience collector
func NewSimpleCollector(maxSize int, gameID string, logger zerolog.Logger) *SimpleCollector {
	return &SimpleCollector{
		buffer:     NewBuffer(maxSize, logger),
		gameID:     gameID,
		serializer: NewSerializer(),
		logger:     logger.With().Str("component", "experience_collector").Logger(),
	}
}

// OnStateTransition collects experience from a state transition
func (c *SimpleCollector) OnStateTransition(prevState, currState *game.GameState, actions map[int]*game.Action) {

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

		// Add to buffer (ring buffer will handle overflow automatically)
		if err := c.buffer.Add(exp); err != nil {
			c.logger.Error().
				Err(err).
				Str("experience_id", expID).
				Msg("Failed to add experience to buffer")
			continue
		}

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
	stats := c.buffer.Stats()
	c.logger.Info().
		Str("game_id", c.gameID).
		Int("total_experiences", stats.CurrentSize).
		Int64("total_added", stats.TotalAdded).
		Int64("total_dropped", stats.TotalDropped).
		Int("winner", finalState.GetWinner()).
		Int("final_turn", finalState.Turn).
		Msg("Game ended, finalizing experience collection")

	// Could add terminal experience processing here if needed
}

// GetExperiences returns a copy of all collected experiences
func (c *SimpleCollector) GetExperiences() []*experiencepb.Experience {
	return c.buffer.GetAll()
}

// GetExperienceCount returns the current number of experiences
func (c *SimpleCollector) GetExperienceCount() int {
	return c.buffer.Size()
}

// Clear removes all experiences from the buffer
func (c *SimpleCollector) Clear() {
	c.buffer.Clear()
}

// GetLatestExperiences returns the n most recent experiences
func (c *SimpleCollector) GetLatestExperiences(n int) []*experiencepb.Experience {
	// Use GetLatest to get the most recent experiences
	return c.buffer.GetLatest(n)
}

// GetCount returns the current number of experiences (without allocation)
func (c *SimpleCollector) GetCount() int {
	return c.buffer.Size()
}

// Close cleanly shuts down the collector
func (c *SimpleCollector) Close() error {
	return c.buffer.Close()
}
