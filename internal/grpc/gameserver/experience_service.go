package gameserver

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/experience"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
)

// BufferedExperienceCollector collects experiences and writes them to a buffer
type BufferedExperienceCollector struct {
	gameID     string
	buffer     *experience.Buffer
	serializer *experience.Serializer
	logger     zerolog.Logger
}

// OnStateTransition implements game.ExperienceCollector
func (c *BufferedExperienceCollector) OnStateTransition(prevState, currState *game.GameState, actions map[int]*game.Action) {
	// Process experience for each player that took an action
	for playerID, action := range actions {
		if action == nil {
			continue // Skip players who didn't act
		}

		// Calculate reward
		reward := experience.CalculateReward(prevState, currState, playerID)

		// Serialize state to tensor
		stateData := c.serializer.StateToTensor(prevState, playerID)
		nextStateData := c.serializer.StateToTensor(currState, playerID)

		// Convert action to index
		actionIndex := int32(0)
		if action != nil && action.Type == game.ActionTypeMove {
			actionIndex = int32(c.serializer.ActionToIndex(action, prevState.Board.W))
		}

		// Determine if game is over for this player
		done := false
		if currState.IsGameOver() {
			done = true
		} else if playerID < len(currState.Players) && !currState.Players[playerID].Alive {
			done = true
		}

		// Create experience
		exp := &experiencepb.Experience{
			ExperienceId: uuid.New().String(),
			GameId:       c.gameID,
			PlayerId:     int32(playerID),
			Turn:         int32(prevState.Turn),
			State: &experiencepb.TensorState{
				Data:  stateData,
				Shape: c.serializer.GetTensorShape(prevState.Board.W, prevState.Board.H),
			},
			Action: actionIndex,
			NextState: &experiencepb.TensorState{
				Data:  nextStateData,
				Shape: c.serializer.GetTensorShape(prevState.Board.W, prevState.Board.H),
			},
			Reward: reward,
			Done:   done,
			Metadata: map[string]string{
				"game_id":   c.gameID,
				"player_id": fmt.Sprintf("%d", playerID),
				"turn":      fmt.Sprintf("%d", prevState.Turn),
			},
		}

		// Add to buffer
		if err := c.buffer.Add(exp); err != nil {
			c.logger.Error().
				Err(err).
				Str("experience_id", exp.ExperienceId).
				Int32("player_id", exp.PlayerId).
				Int32("turn", exp.Turn).
				Msg("Failed to add experience to buffer")
		}
	}
}

// OnGameEnd implements game.ExperienceCollector
func (c *BufferedExperienceCollector) OnGameEnd(finalState *game.GameState) {
	c.logger.Info().
		Str("game_id", c.gameID).
		Int("turn", finalState.Turn).
		Bool("game_over", finalState.IsGameOver()).
		Msg("Game ended, experience collection complete")
}

// ExperienceService implements the ExperienceService gRPC server
type ExperienceService struct {
	experiencepb.UnimplementedExperienceServiceServer

	// Buffer manager for experience storage
	bufferManager *experience.BufferManager

	// Stream aggregator for multi-game streaming
	streamAggregator *StreamAggregator

	// Active streams
	mu            sync.RWMutex
	activeStreams map[string]*experienceStream

	// Configuration
	defaultBufferSize int
	streamTimeout     time.Duration

	// Metrics
	totalBatchesSent     int64
	totalExperiencesSent int64
}

// experienceStream represents an active experience stream
type experienceStream struct {
	id         string
	gameIDs    map[string]bool
	playerIDs  map[int32]bool
	batchSize  int32
	follow     bool
	cancelFunc context.CancelFunc
	lastSent   time.Time
}

// NewExperienceService creates a new experience service
func NewExperienceService(bufferManager *experience.BufferManager) *ExperienceService {
	if bufferManager == nil {
		// Create default buffer manager if none provided
		bufferManager = experience.NewBufferManager(10000, log.Logger)
	}

	return &ExperienceService{
		bufferManager:     bufferManager,
		streamAggregator:  NewStreamAggregator(bufferManager),
		activeStreams:     make(map[string]*experienceStream),
		defaultBufferSize: 10000,
		streamTimeout:     30 * time.Minute,
	}
}

// StreamExperiences streams experiences to clients
func (s *ExperienceService) StreamExperiences(
	req *experiencepb.StreamExperiencesRequest,
	stream experiencepb.ExperienceService_StreamExperiencesServer,
) error {
	// Validate request
	if err := s.validateStreamRequest(req); err != nil {
		return status.Errorf(codes.InvalidArgument, "invalid request: %v", err)
	}

	// Create stream context
	ctx, cancel := context.WithCancel(stream.Context())
	defer cancel()

	// Create stream info
	streamID := uuid.New().String()
	streamInfo := &experienceStream{
		id:         streamID,
		gameIDs:    make(map[string]bool),
		playerIDs:  make(map[int32]bool),
		batchSize:  req.BatchSize,
		follow:     req.Follow,
		cancelFunc: cancel,
		lastSent:   time.Now(),
	}

	// Convert game IDs to map for efficient lookup
	for _, gameID := range req.GameIds {
		streamInfo.gameIDs[gameID] = true
	}

	// Convert player IDs to map for efficient lookup
	for _, playerID := range req.PlayerIds {
		streamInfo.playerIDs[playerID] = true
	}

	// Register stream
	s.registerStream(streamID, streamInfo)
	defer s.unregisterStream(streamID)

	log.Info().
		Str("stream_id", streamID).
		Int("game_ids", len(req.GameIds)).
		Int("player_ids", len(req.PlayerIds)).
		Int32("batch_size", req.BatchSize).
		Bool("follow", req.Follow).
		Msg("Started experience stream")

	// Create merged stream from relevant buffers
	merger := experience.NewStreamMerger(1000, log.Logger)

	// Add sources based on filters
	if len(req.GameIds) > 0 {
		// Stream from specific game buffers
		for _, gameID := range req.GameIds {
			if buffer, exists := s.bufferManager.GetBuffer(gameID); exists {
				merger.AddSource(buffer.StreamChannel())
			}
		}
	} else {
		// Stream from all buffers
		for _, buffer := range s.bufferManager.GetAllBuffers() {
			merger.AddSource(buffer.StreamChannel())
		}
	}

	// Start merging
	merger.Start()
	defer merger.Close()

	// Stream experiences
	batch := make([]*experiencepb.Experience, 0, req.BatchSize)
	batchTimer := time.NewTimer(100 * time.Millisecond)
	defer batchTimer.Stop()

	for {
		select {
		case <-ctx.Done():
			// Context cancelled
			return nil

		case exp, ok := <-merger.Output():
			if !ok {
				// Channel closed
				if !req.Follow {
					// Send final batch if any
					if len(batch) > 0 {
						if err := s.sendBatch(stream, batch); err != nil {
							return err
						}
					}
					return nil
				}
				// In follow mode, continue waiting
				continue
			}

			// Filter experience if needed
			if !s.shouldIncludeExperience(exp, streamInfo) {
				continue
			}

			// Add to batch
			batch = append(batch, exp)

			// Send batch if full
			if int32(len(batch)) >= req.BatchSize {
				if err := s.sendBatch(stream, batch); err != nil {
					return err
				}
				batch = batch[:0]
				batchTimer.Reset(100 * time.Millisecond)
			}

		case <-batchTimer.C:
			// Send partial batch on timeout
			if len(batch) > 0 {
				if err := s.sendBatch(stream, batch); err != nil {
					return err
				}
				batch = batch[:0]
			}
			batchTimer.Reset(100 * time.Millisecond)

		case <-time.After(s.streamTimeout):
			// Stream timeout
			return status.Error(codes.DeadlineExceeded, "stream timeout")
		}
	}
}

// StreamExperienceBatches streams batched experiences using the StreamAggregator
func (s *ExperienceService) StreamExperienceBatches(
	req *experiencepb.StreamExperiencesRequest,
	stream experiencepb.ExperienceService_StreamExperienceBatchesServer,
) error {
	// Validate request
	if err := s.validateStreamRequest(req); err != nil {
		return status.Errorf(codes.InvalidArgument, "invalid request: %v", err)
	}

	// Create aggregated stream
	batchCh, streamID, err := s.streamAggregator.CreateStream(
		stream.Context(),
		req.GameIds,
		req.PlayerIds,
		req.BatchSize,
		req.Follow,
	)
	if err != nil {
		return status.Errorf(codes.Internal, "failed to create stream: %v", err)
	}
	defer s.streamAggregator.CloseStream(streamID)

	log.Info().
		Str("stream_id", streamID).
		Int("game_filters", len(req.GameIds)).
		Int("player_filters", len(req.PlayerIds)).
		Int32("batch_size", req.BatchSize).
		Bool("follow", req.Follow).
		Bool("compression", req.EnableCompression).
		Msg("Started batched experience stream")

	// Stream batches
	var batchCounter int32
	for {
		select {
		case <-stream.Context().Done():
			return nil

		case batch, ok := <-batchCh:
			if !ok {
				// Channel closed
				log.Info().
					Str("stream_id", streamID).
					Int32("total_batches", batchCounter).
					Msg("Stream completed")
				return nil
			}

			if len(batch) == 0 {
				continue
			}

			// Create batch message
			batchMsg := &experiencepb.ExperienceBatch{
				Experiences: batch,
				BatchId:     atomic.AddInt32(&batchCounter, 1),
				StreamId:    streamID,
				CreatedAt:   timestamppb.Now(),
				Metadata: map[string]string{
					"batch_size": fmt.Sprintf("%d", len(batch)),
				},
			}

			// Apply compression if requested
			if req.EnableCompression {
				// TODO: Implement compression
				batchMsg.Metadata["compression"] = "none"
			}

			// Send batch
			if err := stream.Send(batchMsg); err != nil {
				log.Error().
					Err(err).
					Str("stream_id", streamID).
					Int32("batch_id", batchMsg.BatchId).
					Msg("Failed to send batch")
				return err
			}

			// Update metrics
			atomic.AddInt64(&s.totalBatchesSent, 1)
			atomic.AddInt64(&s.totalExperiencesSent, int64(len(batch)))

		case <-time.After(30 * time.Second):
			// Health check timeout
			if !req.Follow {
				// Not in follow mode, timeout
				return status.Error(codes.DeadlineExceeded, "stream timeout")
			}
		}
	}
}

// SubmitExperiences allows clients to submit experiences
func (s *ExperienceService) SubmitExperiences(
	ctx context.Context,
	req *experiencepb.SubmitExperiencesRequest,
) (*experiencepb.SubmitExperiencesResponse, error) {
	// Validate request
	if len(req.Experiences) == 0 {
		return nil, status.Error(codes.InvalidArgument, "no experiences provided")
	}

	if len(req.Experiences) > 1000 {
		return nil, status.Error(codes.InvalidArgument, "too many experiences (max 1000)")
	}

	// Get or create buffer for the game
	gameID := ""
	if len(req.Experiences) > 0 {
		// Extract game ID from first experience
		gameID = req.Experiences[0].GameId
	}

	if gameID == "" {
		return nil, status.Error(codes.InvalidArgument, "game ID required")
	}

	buffer := s.bufferManager.GetOrCreateBuffer(gameID)

	// Add experiences to buffer
	accepted := 0
	rejected := 0

	for _, exp := range req.Experiences {
		// Validate experience
		if err := s.validateExperience(exp); err != nil {
			log.Debug().
				Err(err).
				Str("game_id", exp.GameId).
				Int32("player_id", exp.PlayerId).
				Msg("Rejected invalid experience")
			rejected++
			continue
		}

		// Add to buffer
		if err := buffer.Add(exp); err != nil {
			if errors.Is(err, experience.ErrBufferFull) {
				log.Warn().
					Str("game_id", gameID).
					Msg("Experience buffer full")
			}
			rejected++
			continue
		}

		accepted++
	}

	log.Info().
		Str("game_id", gameID).
		Int("accepted", accepted).
		Int("rejected", rejected).
		Msg("Submitted experiences")

	return &experiencepb.SubmitExperiencesResponse{
		Accepted: int32(accepted),
		Rejected: int32(rejected),
	}, nil
}

// GetExperienceStats returns statistics about experience collection
func (s *ExperienceService) GetExperienceStats(
	ctx context.Context,
	req *experiencepb.GetExperienceStatsRequest,
) (*experiencepb.GetExperienceStatsResponse, error) {
	// Create response
	resp := &experiencepb.GetExperienceStatsResponse{
		TotalExperiences: 0,
		TotalGames:       0,
	}

	// Aggregate stats from all buffers
	totalSize := int64(0)
	totalCapacity := int64(0)
	totalAdded := int64(0)
	totalDropped := int64(0)
	totalStreamed := int64(0)

	buffers := s.bufferManager.GetAllBuffers()

	// Filter by game IDs if specified
	if len(req.GameIds) > 0 {
		gameIDSet := make(map[string]bool)
		for _, id := range req.GameIds {
			gameIDSet[id] = true
		}

		filtered := make(map[string]*experience.Buffer)
		for key, buffer := range buffers {
			if gameIDSet[key] {
				filtered[key] = buffer
			}
		}
		buffers = filtered
	}

	// Collect stats
	for _, buffer := range buffers {
		bufferStats := buffer.Stats()
		totalSize += int64(bufferStats.CurrentSize)
		totalCapacity += int64(bufferStats.Capacity)
		totalAdded += bufferStats.TotalAdded
		totalDropped += bufferStats.TotalDropped
		totalStreamed += bufferStats.TotalStreamed
	}

	resp.TotalExperiences = totalAdded
	resp.TotalGames = int64(len(buffers))

	// TODO: Add more detailed statistics
	// - experiences_per_game
	// - experiences_per_player
	// - average_reward
	// - min/max rewards
	// - oldest/newest experience timestamps

	return resp, nil
}

// Helper methods

func (s *ExperienceService) validateStreamRequest(req *experiencepb.StreamExperiencesRequest) error {
	if req.BatchSize <= 0 {
		req.BatchSize = 32 // Default batch size
	}

	if req.BatchSize > 1000 {
		return fmt.Errorf("batch size too large (max 1000)")
	}

	return nil
}

func (s *ExperienceService) validateExperience(exp *experiencepb.Experience) error {
	if exp == nil {
		return fmt.Errorf("experience is nil")
	}

	if exp.GameId == "" {
		return fmt.Errorf("game ID is required")
	}

	if exp.State == nil || exp.NextState == nil {
		return fmt.Errorf("state and next_state are required")
	}

	if len(exp.State.Data) == 0 || len(exp.NextState.Data) == 0 {
		return fmt.Errorf("state data cannot be empty")
	}

	return nil
}

func (s *ExperienceService) shouldIncludeExperience(exp *experiencepb.Experience, stream *experienceStream) bool {
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

func (s *ExperienceService) sendBatch(
	stream experiencepb.ExperienceService_StreamExperiencesServer,
	batch []*experiencepb.Experience,
) error {
	for _, exp := range batch {
		if err := stream.Send(exp); err != nil {
			return err
		}
	}
	return nil
}

func (s *ExperienceService) registerStream(id string, stream *experienceStream) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activeStreams[id] = stream
}

func (s *ExperienceService) unregisterStream(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.activeStreams, id)
}

// GetBufferManager returns the buffer manager (useful for integration)
func (s *ExperienceService) GetBufferManager() *experience.BufferManager {
	return s.bufferManager
}

// CreateCollector creates an experience collector for a specific game
func (s *ExperienceService) CreateCollector(gameID string) *BufferedExperienceCollector {
	// Create or get buffer for this game
	buffer := s.bufferManager.GetOrCreateBuffer(gameID)

	// Create and return collector that writes to the buffer
	return &BufferedExperienceCollector{
		gameID:     gameID,
		buffer:     buffer,
		serializer: experience.NewSerializer(),
		logger:     log.Logger.With().Str("component", "buffered_collector").Str("game_id", gameID).Logger(),
	}
}
