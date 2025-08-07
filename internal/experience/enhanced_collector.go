package experience

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// CollectorConfig contains configuration for the enhanced collector
type CollectorConfig struct {
	// Buffer configuration
	BufferCapacity int
	BatchSize      int
	FlushInterval  time.Duration

	// Persistence configuration
	PersistenceConfig PersistenceConfig

	// Metrics configuration
	MetricsEnabled  bool
	MetricsInterval time.Duration
}

// DefaultCollectorConfig returns a default collector configuration
func DefaultCollectorConfig() CollectorConfig {
	return CollectorConfig{
		BufferCapacity:    10000,
		BatchSize:         100,
		FlushInterval:     30 * time.Second,
		PersistenceConfig: DefaultPersistenceConfig(),
		MetricsEnabled:    true,
		MetricsInterval:   1 * time.Minute,
	}
}

// EnhancedCollector implements an experience collector with ring buffer, persistence, and batching
type EnhancedCollector struct {
	config      CollectorConfig
	gameID      string
	buffer      *Buffer
	serializer  *Serializer
	persistence PersistenceLayer
	logger      zerolog.Logger

	// Batching
	batchChan  chan []*experiencepb.Experience
	flushTimer *time.Timer

	// Lifecycle
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	closeChan chan struct{}

	// Metrics
	mu      sync.RWMutex
	metrics CollectorMetrics
}

// CollectorMetrics contains metrics for the collector
type CollectorMetrics struct {
	ExperiencesCollected int64
	ExperiencesPersisted int64
	ExperiencesDropped   int64
	BatchesFlushed       int64
	PersistenceErrors    int64
	OverflowEvents       int64
	LastCollectionTime   time.Time
	LastPersistenceTime  time.Time
}

// NewEnhancedCollector creates a new enhanced experience collector
func NewEnhancedCollector(config CollectorConfig, gameID string, logger zerolog.Logger) (*EnhancedCollector, error) {
	// Create buffer
	buffer := NewBuffer(config.BufferCapacity, logger)

	// Create persistence layer
	persistence, err := NewPersistenceLayer(config.PersistenceConfig, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create persistence layer: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	ec := &EnhancedCollector{
		config:      config,
		gameID:      gameID,
		buffer:      buffer,
		serializer:  NewSerializer(),
		persistence: persistence,
		logger:      logger.With().Str("component", "enhanced_collector").Logger(),
		batchChan:   make(chan []*experiencepb.Experience, 10),
		closeChan:   make(chan struct{}),
		ctx:         ctx,
		cancel:      cancel,
	}

	// Start background workers
	ec.wg.Add(2)
	go ec.batchProcessor()
	go ec.persistenceWorker()

	// Start metrics reporter if enabled
	if config.MetricsEnabled {
		ec.wg.Add(1)
		go ec.metricsReporter()
	}

	ec.logger.Info().
		Str("game_id", gameID).
		Int("buffer_capacity", config.BufferCapacity).
		Str("overflow_strategy", string(config.PersistenceConfig.OverflowStrategy)).
		Msg("Enhanced collector initialized")

	return ec, nil
}

// OnStateTransition collects experience from a state transition
func (ec *EnhancedCollector) OnStateTransition(prevState, currState *game.GameState, actions map[int]*game.Action) {
	ec.mu.Lock()
	ec.metrics.LastCollectionTime = time.Now()
	ec.mu.Unlock()

	// Process experience for each player that took an action
	for playerID, action := range actions {
		if action == nil {
			continue
		}

		// Generate experience
		exp := ec.createExperience(prevState, currState, playerID, action)

		// Handle buffer overflow based on strategy
		if ec.buffer.IsFull() {
			if ec.config.PersistenceConfig.OverflowStrategy == OverflowStrategyDropNewest {
				// Drop newest strategy - don't add new experiences
				ec.mu.Lock()
				ec.metrics.ExperiencesDropped++
				ec.metrics.OverflowEvents++
				ec.mu.Unlock()

				ec.logger.Debug().
					Str("experience_id", exp.ExperienceId).
					Msg("Dropped new experience due to buffer overflow")
				continue
			}
			ec.handleOverflow()
		}

		// Add to buffer
		if err := ec.buffer.Add(exp); err != nil {
			ec.logger.Error().
				Err(err).
				Str("experience_id", exp.ExperienceId).
				Msg("Failed to add experience to buffer")
			continue
		}

		ec.mu.Lock()
		ec.metrics.ExperiencesCollected++
		ec.mu.Unlock()

		ec.logger.Debug().
			Str("experience_id", exp.ExperienceId).
			Int("player_id", playerID).
			Int("turn", currState.Turn).
			Float32("reward", exp.Reward).
			Bool("done", exp.Done).
			Msg("Collected experience")
	}
}

// createExperience creates an experience protobuf from game states
func (ec *EnhancedCollector) createExperience(prevState, currState *game.GameState, playerID int, action *game.Action) *experiencepb.Experience {
	// Generate unique experience ID
	expID := uuid.New().String()

	// Serialize states from player's perspective
	stateTensor := ec.serializer.StateToTensor(prevState, playerID)
	nextStateTensor := ec.serializer.StateToTensor(currState, playerID)

	// Calculate reward
	reward := CalculateReward(prevState, currState, playerID)

	// Generate action mask for legal moves
	actionMask := ec.serializer.GenerateActionMask(prevState, playerID)

	// Convert action to flattened index
	actionIndex := ec.serializer.ActionToIndex(action, prevState.Board.W)

	// Check if game ended
	done := currState.IsGameOver()

	return &experiencepb.Experience{
		ExperienceId: expID,
		GameId:       ec.gameID,
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
			"collector_version": "2.0.0",
			"collector_type":    "enhanced",
		},
	}
}

// handleOverflow handles buffer overflow based on configured strategy
func (ec *EnhancedCollector) handleOverflow() {
	ec.mu.Lock()
	ec.metrics.OverflowEvents++
	ec.mu.Unlock()

	switch ec.config.PersistenceConfig.OverflowStrategy {
	case OverflowStrategyPersist:
		// Flush some experiences to persistence
		experiences := ec.buffer.Get(ec.config.BatchSize)
		if len(experiences) > 0 {
			select {
			case ec.batchChan <- experiences:
				ec.logger.Debug().
					Int("batch_size", len(experiences)).
					Msg("Flushed experiences due to overflow")
			default:
				// Batch channel full, drop experiences
				ec.mu.Lock()
				ec.metrics.ExperiencesDropped += int64(len(experiences))
				ec.mu.Unlock()
			}
		}

	case OverflowStrategyDropOldest:
		// Ring buffer already handles this by default
		ec.logger.Debug().Msg("Buffer overflow - dropping oldest experiences")

	case OverflowStrategyDropNewest:
		// Handled in OnStateTransition
		ec.logger.Debug().Msg("Buffer overflow - will drop new experiences")
	}
}

// batchProcessor processes batches of experiences
func (ec *EnhancedCollector) batchProcessor() {
	defer ec.wg.Done()

	batch := make([]*experiencepb.Experience, 0, ec.config.BatchSize)
	flushTimer := time.NewTimer(ec.config.FlushInterval)
	defer flushTimer.Stop()

	for {
		select {
		case <-ec.ctx.Done():
			// Flush remaining batch
			if len(batch) > 0 {
				ec.batchChan <- batch
			}
			close(ec.batchChan)
			return

		case exp := <-ec.buffer.StreamChannel():
			batch = append(batch, exp)

			if len(batch) >= ec.config.BatchSize {
				// Batch full, send it
				select {
				case ec.batchChan <- batch:
					batch = make([]*experiencepb.Experience, 0, ec.config.BatchSize)
					flushTimer.Reset(ec.config.FlushInterval)
				case <-ec.ctx.Done():
					return
				}
			}

		case <-flushTimer.C:
			// Flush partial batch
			if len(batch) > 0 {
				select {
				case ec.batchChan <- batch:
					batch = make([]*experiencepb.Experience, 0, ec.config.BatchSize)
				case <-ec.ctx.Done():
					return
				}
			}
			flushTimer.Reset(ec.config.FlushInterval)
		}
	}
}

// persistenceWorker handles persistence of experience batches
func (ec *EnhancedCollector) persistenceWorker() {
	defer ec.wg.Done()

	for batch := range ec.batchChan {
		if len(batch) == 0 {
			continue
		}

		// Persist batch
		if err := ec.persistence.Write(ec.ctx, batch); err != nil {
			ec.mu.Lock()
			ec.metrics.PersistenceErrors++
			ec.mu.Unlock()

			ec.logger.Error().
				Err(err).
				Int("batch_size", len(batch)).
				Msg("Failed to persist experience batch")

			// Return experiences to buffer if possible
			if ec.config.PersistenceConfig.OverflowStrategy != OverflowStrategyDropOldest {
				if err := ec.buffer.AddBatch(batch); err != nil {
					ec.mu.Lock()
					ec.metrics.ExperiencesDropped += int64(len(batch))
					ec.mu.Unlock()
				}
			}
		} else {
			ec.mu.Lock()
			ec.metrics.ExperiencesPersisted += int64(len(batch))
			ec.metrics.BatchesFlushed++
			ec.metrics.LastPersistenceTime = time.Now()
			ec.mu.Unlock()

			ec.logger.Debug().
				Int("batch_size", len(batch)).
				Msg("Persisted experience batch")
		}
	}
}

// metricsReporter periodically reports metrics
func (ec *EnhancedCollector) metricsReporter() {
	defer ec.wg.Done()

	ticker := time.NewTicker(ec.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ec.ctx.Done():
			return

		case <-ticker.C:
			ec.reportMetrics()
		}
	}
}

// reportMetrics logs current metrics
func (ec *EnhancedCollector) reportMetrics() {
	ec.mu.RLock()
	metrics := ec.metrics
	ec.mu.RUnlock()

	bufferStats := ec.buffer.Stats()
	persistenceStats := ec.persistence.Stats()

	ec.logger.Info().
		Int64("experiences_collected", metrics.ExperiencesCollected).
		Int64("experiences_persisted", metrics.ExperiencesPersisted).
		Int64("experiences_dropped", metrics.ExperiencesDropped).
		Int64("batches_flushed", metrics.BatchesFlushed).
		Int64("persistence_errors", metrics.PersistenceErrors).
		Int64("overflow_events", metrics.OverflowEvents).
		Int("buffer_size", bufferStats.CurrentSize).
		Float64("buffer_utilization_pct", bufferStats.UtilizationPct).
		Int64("buffer_dropped", bufferStats.TotalDropped).
		Int64("persistence_written", persistenceStats.TotalWritten).
		Int64("persistence_bytes", persistenceStats.BytesWritten).
		Msg("Collector metrics")
}

// OnGameEnd handles terminal states
func (ec *EnhancedCollector) OnGameEnd(finalState *game.GameState) {
	// Flush all remaining experiences
	remaining := ec.buffer.GetAll()
	if len(remaining) > 0 {
		select {
		case ec.batchChan <- remaining:
			ec.logger.Info().
				Int("final_batch_size", len(remaining)).
				Msg("Flushed final experiences on game end")
		default:
			ec.logger.Warn().
				Int("dropped_count", len(remaining)).
				Msg("Failed to flush final experiences - batch channel full")
		}
	}

	ec.logger.Info().
		Str("game_id", ec.gameID).
		Int("winner", finalState.GetWinner()).
		Int("final_turn", finalState.Turn).
		Int64("total_collected", ec.metrics.ExperiencesCollected).
		Int64("total_persisted", ec.metrics.ExperiencesPersisted).
		Msg("Game ended, finalizing experience collection")
}

// GetExperiences returns experiences from the buffer
func (ec *EnhancedCollector) GetExperiences(n int) []*experiencepb.Experience {
	return ec.buffer.Get(n)
}

// GetAllExperiences returns all experiences from the buffer
func (ec *EnhancedCollector) GetAllExperiences() []*experiencepb.Experience {
	return ec.buffer.GetAll()
}

// GetExperienceCount returns the current number of experiences in the buffer
func (ec *EnhancedCollector) GetExperienceCount() int {
	return ec.buffer.Size()
}

// GetMetrics returns current collector metrics
func (ec *EnhancedCollector) GetMetrics() CollectorMetrics {
	ec.mu.RLock()
	defer ec.mu.RUnlock()
	return ec.metrics
}

// GetBufferStats returns buffer statistics
func (ec *EnhancedCollector) GetBufferStats() BufferStats {
	return ec.buffer.Stats()
}

// GetPersistenceStats returns persistence statistics
func (ec *EnhancedCollector) GetPersistenceStats() PersistenceStats {
	return ec.persistence.Stats()
}

// FlushToPersistence manually flushes buffer to persistence
func (ec *EnhancedCollector) FlushToPersistence() error {
	experiences := ec.buffer.GetAll()
	if len(experiences) == 0 {
		return nil
	}

	if err := ec.persistence.Write(ec.ctx, experiences); err != nil {
		// Return experiences to buffer
		if err := ec.buffer.AddBatch(experiences); err != nil {
			ec.logger.Error().
				Err(err).
				Int("count", len(experiences)).
				Msg("Failed to return experiences to buffer after persistence failure")
		}
		return fmt.Errorf("failed to persist experiences: %w", err)
	}

	ec.mu.Lock()
	ec.metrics.ExperiencesPersisted += int64(len(experiences))
	ec.metrics.BatchesFlushed++
	ec.mu.Unlock()

	return nil
}

// LoadFromPersistence loads experiences from persistence into buffer
func (ec *EnhancedCollector) LoadFromPersistence(limit int) error {
	experiences, err := ec.persistence.Read(ec.ctx, ec.gameID, limit)
	if err != nil {
		return fmt.Errorf("failed to read from persistence: %w", err)
	}

	if err := ec.buffer.AddBatch(experiences); err != nil {
		return fmt.Errorf("failed to add experiences to buffer: %w", err)
	}

	ec.logger.Info().
		Int("loaded_count", len(experiences)).
		Msg("Loaded experiences from persistence")

	return nil
}

// Close cleanly shuts down the collector
func (ec *EnhancedCollector) Close() error {
	ec.logger.Info().Msg("Closing enhanced collector")

	// Cancel context to stop workers
	ec.cancel()

	// Wait for workers to finish
	ec.wg.Wait()

	// Final metrics report
	ec.reportMetrics()

	// Close buffer and persistence
	if err := ec.buffer.Close(); err != nil {
		ec.logger.Error().Err(err).Msg("Failed to close buffer")
	}

	if err := ec.persistence.Close(); err != nil {
		ec.logger.Error().Err(err).Msg("Failed to close persistence layer")
	}

	close(ec.closeChan)

	return nil
}
