package gameserver

import (
	"context"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/experience"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
)

// StreamAggregator aggregates experience streams from multiple games
// and provides unified streaming with batching and backpressure handling
type StreamAggregator struct {
	// Buffer manager for accessing game buffers
	bufferManager *experience.BufferManager

	// Active stream subscriptions
	mu      sync.RWMutex
	streams map[string]*aggregatedStream

	// Configuration
	maxBatchSize      int32
	batchTimeout      time.Duration
	maxPendingBatches int

	// Metrics
	totalStreams      int64
	totalExperiences  int64
	droppedExperiences int64

	logger zerolog.Logger
}

// aggregatedStream represents a unified stream from multiple games
type aggregatedStream struct {
	id        string
	ctx       context.Context
	cancel    context.CancelFunc
	
	// Filters
	gameIDs   map[string]bool
	playerIDs map[int32]bool
	
	// Output channel for batched experiences
	outputCh  chan []*experiencepb.Experience
	
	// Stream merger for combining multiple game streams
	merger    *experience.StreamMerger
	
	// Batch processor
	batchProcessor *BatchProcessor
	
	// Stats
	experiencesSent int64
	lastActivity    time.Time
}

// NewStreamAggregator creates a new stream aggregator
func NewStreamAggregator(bufferManager *experience.BufferManager) *StreamAggregator {
	return &StreamAggregator{
		bufferManager:     bufferManager,
		streams:          make(map[string]*aggregatedStream),
		maxBatchSize:     32,
		batchTimeout:     100 * time.Millisecond,
		maxPendingBatches: 100,
		logger:           log.Logger.With().Str("component", "stream_aggregator").Logger(),
	}
}

// CreateStream creates a new aggregated stream based on filters
func (sa *StreamAggregator) CreateStream(
	ctx context.Context,
	gameIDs []string,
	playerIDs []int32,
	batchSize int32,
	follow bool,
) (<-chan []*experiencepb.Experience, string, error) {
	sa.mu.Lock()
	defer sa.mu.Unlock()

	// Create stream context
	streamCtx, cancel := context.WithCancel(ctx)
	
	// Create stream ID
	streamID := generateStreamID()
	
	// Create output channel
	outputCh := make(chan []*experiencepb.Experience, sa.maxPendingBatches)
	
	// Create stream merger
	merger := experience.NewStreamMerger(1000, sa.logger)
	
	// Create batch processor
	batchProcessor := NewBatchProcessor(batchSize, sa.batchTimeout)
	
	// Create aggregated stream
	stream := &aggregatedStream{
		id:             streamID,
		ctx:            streamCtx,
		cancel:         cancel,
		gameIDs:        make(map[string]bool),
		playerIDs:      make(map[int32]bool),
		outputCh:       outputCh,
		merger:         merger,
		batchProcessor: batchProcessor,
		lastActivity:   time.Now(),
	}
	
	// Set up filters
	for _, gameID := range gameIDs {
		stream.gameIDs[gameID] = true
	}
	for _, playerID := range playerIDs {
		stream.playerIDs[playerID] = true
	}
	
	// Add sources to merger
	if len(gameIDs) > 0 {
		// Stream from specific games
		for _, gameID := range gameIDs {
			if buffer, exists := sa.bufferManager.GetBuffer(gameID); exists {
				merger.AddSource(buffer.StreamChannel())
			}
		}
	} else {
		// Stream from all games
		for _, buffer := range sa.bufferManager.GetAllBuffers() {
			merger.AddSource(buffer.StreamChannel())
		}
	}
	
	// Start merger
	merger.Start()
	
	// Store stream
	sa.streams[streamID] = stream
	sa.totalStreams++
	
	// Start processing goroutine
	go sa.processStream(stream, follow)
	
	sa.logger.Info().
		Str("stream_id", streamID).
		Int("game_filters", len(gameIDs)).
		Int("player_filters", len(playerIDs)).
		Int32("batch_size", batchSize).
		Bool("follow", follow).
		Msg("Created aggregated stream")
	
	return outputCh, streamID, nil
}

// processStream processes experiences from merger and sends batched output
func (sa *StreamAggregator) processStream(stream *aggregatedStream, follow bool) {
	defer func() {
		stream.cancel()
		stream.merger.Close()
		close(stream.outputCh)
		sa.removeStream(stream.id)
	}()
	
	// Start batch processor
	batchCh := stream.batchProcessor.Start(stream.ctx)
	
	// Process experiences
	go func() {
		for {
			select {
			case <-stream.ctx.Done():
				return
				
			case exp, ok := <-stream.merger.Output():
				if !ok {
					if !follow {
						// Merger closed and not following, stop
						stream.batchProcessor.Flush()
						return
					}
					// In follow mode, continue waiting
					continue
				}
				
				// Filter experience
				if !sa.shouldIncludeExperience(exp, stream) {
					continue
				}
				
				// Add to batch processor
				if !stream.batchProcessor.Add(exp) {
					sa.droppedExperiences++
					sa.logger.Warn().
						Str("stream_id", stream.id).
						Msg("Dropped experience due to backpressure")
				}
				
				stream.lastActivity = time.Now()
			}
		}
	}()
	
	// Send batches to output channel
	for {
		select {
		case <-stream.ctx.Done():
			return
			
		case batch, ok := <-batchCh:
			if !ok {
				return
			}
			
			if len(batch) == 0 {
				continue
			}
			
			// Send batch with backpressure handling
			select {
			case stream.outputCh <- batch:
				stream.experiencesSent += int64(len(batch))
				sa.totalExperiences += int64(len(batch))
				
			case <-time.After(5 * time.Second):
				// Consumer too slow, drop batch
				sa.droppedExperiences += int64(len(batch))
				sa.logger.Warn().
					Str("stream_id", stream.id).
					Int("batch_size", len(batch)).
					Msg("Dropped batch due to slow consumer")
				
			case <-stream.ctx.Done():
				return
			}
		}
	}
}

// shouldIncludeExperience checks if an experience matches stream filters
func (sa *StreamAggregator) shouldIncludeExperience(
	exp *experiencepb.Experience,
	stream *aggregatedStream,
) bool {
	// Filter by game ID if specified
	if len(stream.gameIDs) > 0 && !stream.gameIDs[exp.GameId] {
		return false
	}
	
	// Filter by player ID if specified
	if len(stream.playerIDs) > 0 && !stream.playerIDs[exp.PlayerId] {
		return false
	}
	
	return true
}

// CloseStream closes an active stream
func (sa *StreamAggregator) CloseStream(streamID string) {
	sa.mu.Lock()
	defer sa.mu.Unlock()
	
	if stream, exists := sa.streams[streamID]; exists {
		stream.cancel()
		sa.logger.Info().
			Str("stream_id", streamID).
			Int64("experiences_sent", stream.experiencesSent).
			Msg("Closed stream")
	}
}

// removeStream removes a stream from the registry
func (sa *StreamAggregator) removeStream(streamID string) {
	sa.mu.Lock()
	defer sa.mu.Unlock()
	delete(sa.streams, streamID)
}

// GetStats returns aggregator statistics
func (sa *StreamAggregator) GetStats() map[string]interface{} {
	sa.mu.RLock()
	defer sa.mu.RUnlock()
	
	activeStreams := make([]map[string]interface{}, 0, len(sa.streams))
	for id, stream := range sa.streams {
		activeStreams = append(activeStreams, map[string]interface{}{
			"id":               id,
			"experiences_sent": stream.experiencesSent,
			"last_activity":    stream.lastActivity,
		})
	}
	
	return map[string]interface{}{
		"active_streams":      len(sa.streams),
		"total_streams":       sa.totalStreams,
		"total_experiences":   sa.totalExperiences,
		"dropped_experiences": sa.droppedExperiences,
		"streams":            activeStreams,
	}
}

// generateStreamID generates a unique stream ID
func generateStreamID() string {
	return "stream-" + time.Now().Format("20060102-150405") + "-" + generateRandomID(8)
}

// generateRandomID generates a random ID of specified length
func generateRandomID(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
	}
	return string(b)
}