package gameserver

import (
	"context"
	"sync"
	"time"

	"github.com/rs/zerolog"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/experience"
)

// ExperienceAggregator centralizes experience collection from all games
// to reduce goroutine proliferation
type ExperienceAggregator struct {
	// Map of game ID to collector
	collectors map[string]*experience.SimpleCollector
	mu         sync.RWMutex

	// Channel for game IDs that need collection
	updateCh chan string

	// Buffer manager for storing experiences
	bufferManager *experience.BufferManager

	// Background context for the aggregator
	ctx    context.Context
	cancel context.CancelFunc

	// Logger
	logger zerolog.Logger

	// Configuration
	batchSize     int
	flushInterval time.Duration
	maxBatchWait  time.Duration
}

// NewExperienceAggregator creates a new centralized experience aggregator
func NewExperienceAggregator(bufferManager *experience.BufferManager, logger zerolog.Logger) *ExperienceAggregator {
	ctx, cancel := context.WithCancel(context.Background())

	return &ExperienceAggregator{
		collectors:    make(map[string]*experience.SimpleCollector),
		updateCh:      make(chan string, 1000), // Buffered to avoid blocking
		bufferManager: bufferManager,
		ctx:           ctx,
		cancel:        cancel,
		logger:        logger.With().Str("component", "experience_aggregator").Logger(),
		batchSize:     100,
		flushInterval: 50 * time.Millisecond, // More frequent than per-game 100ms
		maxBatchWait:  10 * time.Millisecond, // Max time to wait for batch
	}
}

// Start begins the background collection process
func (ea *ExperienceAggregator) Start() {
	go ea.runCollector()
}

// Stop gracefully shuts down the aggregator
func (ea *ExperienceAggregator) Stop() {
	ea.cancel()
}

// RegisterGame registers a new game with the aggregator
func (ea *ExperienceAggregator) RegisterGame(gameID string, collector *experience.SimpleCollector) {
	ea.mu.Lock()
	defer ea.mu.Unlock()

	ea.collectors[gameID] = collector
	ea.logger.Debug().
		Str("game_id", gameID).
		Int("total_games", len(ea.collectors)).
		Msg("Registered game with aggregator")
}

// UnregisterGame removes a game from the aggregator
func (ea *ExperienceAggregator) UnregisterGame(gameID string) {
	ea.mu.Lock()
	defer ea.mu.Unlock()

	// Process any remaining experiences before removing
	if collector, exists := ea.collectors[gameID]; exists {
		ea.processGameExperiences(gameID, collector)
		delete(ea.collectors, gameID)

		ea.logger.Debug().
			Str("game_id", gameID).
			Int("remaining_games", len(ea.collectors)).
			Msg("Unregistered game from aggregator")
	}
}

// NotifyExperienceAvailable signals that a game has new experiences
// This is non-blocking due to buffered channel
func (ea *ExperienceAggregator) NotifyExperienceAvailable(gameID string) {
	select {
	case ea.updateCh <- gameID:
		// Successfully queued
	default:
		// Channel full, will be picked up in next batch anyway
		ea.logger.Warn().
			Str("game_id", gameID).
			Msg("Update channel full, experience will be collected in next batch")
	}
}

// runCollector is the main background goroutine that collects experiences
func (ea *ExperienceAggregator) runCollector() {
	ticker := time.NewTicker(ea.flushInterval)
	defer ticker.Stop()

	// Track games that need collection
	pendingGames := make(map[string]struct{})

	ea.logger.Info().Msg("Experience aggregator started")

	for {
		select {
		case <-ea.ctx.Done():
			// Final flush before shutdown
			ea.flushAllGames()
			ea.logger.Info().Msg("Experience aggregator stopped")
			return

		case gameID := <-ea.updateCh:
			// Mark game as needing collection
			pendingGames[gameID] = struct{}{}

			// Check if we should process immediately
			if len(pendingGames) >= ea.batchSize {
				ea.processPendingGames(pendingGames)
				pendingGames = make(map[string]struct{})
			}

		case <-ticker.C:
			// Regular interval collection
			if len(pendingGames) > 0 {
				ea.processPendingGames(pendingGames)
				pendingGames = make(map[string]struct{})
			} else {
				// No pending games, do a full scan (less frequently)
				ea.collectBatch()
			}
		}
	}
}

// processPendingGames processes experiences from specific games
func (ea *ExperienceAggregator) processPendingGames(gameIDs map[string]struct{}) {
	ea.mu.RLock()
	defer ea.mu.RUnlock()

	processed := 0
	for gameID := range gameIDs {
		if collector, exists := ea.collectors[gameID]; exists {
			if ea.processGameExperiences(gameID, collector) {
				processed++
			}
		}
	}

	if processed > 0 {
		ea.logger.Debug().
			Int("games_processed", processed).
			Int("batch_size", len(gameIDs)).
			Msg("Processed pending game experiences")
	}
}

// collectBatch processes all games (called less frequently)
func (ea *ExperienceAggregator) collectBatch() {
	ea.mu.RLock()
	defer ea.mu.RUnlock()

	processed := 0
	totalExperiences := 0

	for gameID, collector := range ea.collectors {
		if ea.processGameExperiences(gameID, collector) {
			processed++
			totalExperiences += collector.GetCount() // Get count before clear
		}
	}

	if processed > 0 {
		ea.logger.Debug().
			Int("games_processed", processed).
			Int("total_experiences", totalExperiences).
			Msg("Batch collection completed")
	}
}

// processGameExperiences transfers experiences from a collector to the buffer
func (ea *ExperienceAggregator) processGameExperiences(gameID string, collector *experience.SimpleCollector) bool {
	experiences := collector.GetExperiences()
	if len(experiences) == 0 {
		return false
	}

	// Get or create buffer for this game
	buffer := ea.bufferManager.GetOrCreateBuffer(gameID)

	// Add experiences to buffer
	added := 0
	for _, exp := range experiences {
		if err := buffer.Add(exp); err != nil {
			ea.logger.Error().
				Err(err).
				Str("game_id", gameID).
				Msg("Failed to add experience to buffer")
		} else {
			added++
		}
	}

	// Clear collector after successful transfer
	if added > 0 {
		collector.Clear()
		ea.logger.Debug().
			Str("game_id", gameID).
			Int("experiences_transferred", added).
			Msg("Transferred experiences to buffer")
	}

	return added > 0
}

// flushAllGames processes all remaining experiences (called on shutdown)
func (ea *ExperienceAggregator) flushAllGames() {
	ea.mu.RLock()
	defer ea.mu.RUnlock()

	flushed := 0
	for gameID, collector := range ea.collectors {
		if ea.processGameExperiences(gameID, collector) {
			flushed++
		}
	}

	if flushed > 0 {
		ea.logger.Info().
			Int("games_flushed", flushed).
			Msg("Flushed all remaining experiences on shutdown")
	}
}

// GetStats returns statistics about the aggregator
func (ea *ExperienceAggregator) GetStats() map[string]interface{} {
	ea.mu.RLock()
	defer ea.mu.RUnlock()

	return map[string]interface{}{
		"active_games":     len(ea.collectors),
		"flush_interval":   ea.flushInterval.String(),
		"batch_size":       ea.batchSize,
		"update_queue_len": len(ea.updateCh),
	}
}
