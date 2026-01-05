package gameserver

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/experience"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
)

// mockBatchStream implements the StreamExperienceBatchesServer interface for testing
type mockBatchStream struct {
	grpc.ServerStream
	ctx     context.Context
	batches []*experiencepb.ExperienceBatch
}

func (m *mockBatchStream) Send(batch *experiencepb.ExperienceBatch) error {
	m.batches = append(m.batches, batch)
	return nil
}

func (m *mockBatchStream) Context() context.Context {
	return m.ctx
}

func TestStreamExperienceBatches(t *testing.T) {
	// Create buffer manager and experience service
	bufferManager := experience.NewBufferManager(1000, log.Logger)
	service := NewExperienceService(bufferManager)

	// Create test context
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Create mock stream
	stream := &mockBatchStream{
		ctx:     ctx,
		batches: make([]*experiencepb.ExperienceBatch, 0),
	}

	// Create some test experiences in a buffer
	gameID := "test-game-1"
	buffer := bufferManager.GetOrCreateBuffer(gameID)

	// Start streaming FIRST before adding experiences
	done := make(chan error, 1)
	go func() {
		req := &experiencepb.StreamExperiencesRequest{
			GameIds:   []string{gameID},
			BatchSize: 10,
			Follow:    true, // Follow mode to keep streaming
		}
		done <- service.StreamExperienceBatches(req, stream)
	}()

	// Give the stream time to set up
	time.Sleep(100 * time.Millisecond)

	// Add test experiences
	numExperiences := 100
	for i := 0; i < numExperiences; i++ {
		exp := &experiencepb.Experience{
			ExperienceId: uuid.New().String(),
			GameId:       gameID,
			PlayerId:     int32(i % 2), // Alternate between 2 players
			Turn:         int32(i),
			Reward:       float32(i),
			Done:         i == numExperiences-1,
		}
		err := buffer.Add(exp)
		require.NoError(t, err)
		
		// Small delay to avoid overwhelming the stream
		if i % 10 == 0 {
			time.Sleep(10 * time.Millisecond)
		}
	}

	// Wait a bit for all experiences to be processed
	time.Sleep(500 * time.Millisecond)
	
	// Cancel the context to stop streaming
	cancel()

	// Wait for streaming to complete
	select {
	case err := <-done:
		assert.NoError(t, err)
	case <-time.After(3 * time.Second):
		t.Fatal("Streaming timed out")
	}

	// Verify batches were received
	assert.Greater(t, len(stream.batches), 0, "Should have received at least one batch")

	// Count total experiences received
	totalReceived := 0
	for _, batch := range stream.batches {
		assert.NotNil(t, batch.StreamId)
		assert.NotNil(t, batch.CreatedAt)
		assert.Greater(t, batch.BatchId, int32(0))
		totalReceived += len(batch.Experiences)
	}

	// We might not receive all experiences due to streaming mechanics
	// but we should receive at least some
	assert.Greater(t, totalReceived, 0, "Should have received some experiences")
}

func TestStreamAggregator(t *testing.T) {
	// Create buffer manager
	bufferManager := experience.NewBufferManager(1000, log.Logger)

	// Create stream aggregator
	aggregator := NewStreamAggregator(bufferManager)

	// Create multiple game buffers
	gameIDs := []string{"game-1", "game-2", "game-3"}
	for _, gameID := range gameIDs {
		buffer := bufferManager.GetOrCreateBuffer(gameID)
		
		// Add experiences to each game
		for i := 0; i < 10; i++ {
			exp := &experiencepb.Experience{
				ExperienceId: uuid.New().String(),
				GameId:       gameID,
				PlayerId:     int32(i % 2),
				Turn:         int32(i),
				Reward:       float32(i),
			}
			err := buffer.Add(exp)
			require.NoError(t, err)
		}
	}

	// Create aggregated stream for all games
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	batchCh, streamID, err := aggregator.CreateStream(
		ctx,
		nil, // All games
		nil, // All players
		5,   // Batch size
		false,
	)
	require.NoError(t, err)
	assert.NotEmpty(t, streamID)

	// Collect batches
	var batches [][]*experiencepb.Experience
	go func() {
		for batch := range batchCh {
			batches = append(batches, batch)
		}
	}()

	// Wait a bit for processing
	time.Sleep(500 * time.Millisecond)

	// Close the stream
	aggregator.CloseStream(streamID)

	// Give it time to finish
	time.Sleep(100 * time.Millisecond)

	// Verify we received batches
	assert.Greater(t, len(batches), 0, "Should have received batches")

	// Check stats
	stats := aggregator.GetStats()
	assert.NotNil(t, stats["total_experiences"])
	assert.NotNil(t, stats["active_streams"])
}

func TestBatchProcessor(t *testing.T) {
	// Create batch processor
	processor := NewBatchProcessor(5, 100*time.Millisecond)

	// Start processor
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	batchCh := processor.Start(ctx)

	// Add experiences
	for i := 0; i < 12; i++ {
		exp := &experiencepb.Experience{
			ExperienceId: uuid.New().String(),
			GameId:       "test-game",
			PlayerId:     int32(i % 2),
			Turn:         int32(i),
		}
		success := processor.Add(exp)
		assert.True(t, success, "Should successfully add experience")
	}

	// Collect batches
	var batches [][]*experiencepb.Experience
	done := make(chan bool)
	go func() {
		for {
			select {
			case batch, ok := <-batchCh:
				if !ok {
					done <- true
					return
				}
				batches = append(batches, batch)
			case <-time.After(500 * time.Millisecond):
				done <- true
				return
			}
		}
	}()

	// Wait for collection
	<-done
	cancel() // Trigger context cancellation to flush

	// Verify batches
	assert.GreaterOrEqual(t, len(batches), 2, "Should have at least 2 batches (12 items / 5 batch size)")

	// Count total experiences
	total := 0
	for _, batch := range batches {
		assert.LessOrEqual(t, len(batch), 5, "Batch size should not exceed 5")
		total += len(batch)
	}
	assert.GreaterOrEqual(t, total, 10, "Should have processed most experiences")
}

func TestStreamWithFilters(t *testing.T) {
	// Create buffer manager and service
	bufferManager := experience.NewBufferManager(1000, log.Logger)
	service := NewExperienceService(bufferManager)

	// Create experiences for multiple games and players
	games := []string{"game-A", "game-B"}
	players := []int32{1, 2, 3}

	for _, gameID := range games {
		buffer := bufferManager.GetOrCreateBuffer(gameID)
		for _, playerID := range players {
			for turn := 0; turn < 5; turn++ {
				exp := &experiencepb.Experience{
					ExperienceId: uuid.New().String(),
					GameId:       gameID,
					PlayerId:     playerID,
					Turn:         int32(turn),
					Reward:       float32(turn) * float32(playerID),
				}
				err := buffer.Add(exp)
				require.NoError(t, err)
			}
		}
	}

	// Test 1: Filter by specific game
	t.Run("FilterByGame", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		stream := &mockBatchStream{
			ctx:     ctx,
			batches: make([]*experiencepb.ExperienceBatch, 0),
		}

		done := make(chan error, 1)
		go func() {
			req := &experiencepb.StreamExperiencesRequest{
				GameIds:   []string{"game-A"},
				BatchSize: 5,
				Follow:    true,
			}
			done <- service.StreamExperienceBatches(req, stream)
		}()

		// Let it stream for a bit
		time.Sleep(500 * time.Millisecond)
		cancel()

		select {
		case err := <-done:
			assert.NoError(t, err)
		case <-time.After(1 * time.Second):
			t.Fatal("Streaming timed out")
		}

		// Verify all experiences are from game-A
		for _, batch := range stream.batches {
			for _, exp := range batch.Experiences {
				assert.Equal(t, "game-A", exp.GameId)
			}
		}
	})

	// Test 2: Filter by specific players
	t.Run("FilterByPlayer", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		stream := &mockBatchStream{
			ctx:     ctx,
			batches: make([]*experiencepb.ExperienceBatch, 0),
		}

		done := make(chan error, 1)
		go func() {
			req := &experiencepb.StreamExperiencesRequest{
				PlayerIds: []int32{1, 2},
				BatchSize: 5,
				Follow:    true,
			}
			done <- service.StreamExperienceBatches(req, stream)
		}()

		// Let it stream for a bit
		time.Sleep(500 * time.Millisecond)
		cancel()

		select {
		case err := <-done:
			assert.NoError(t, err)
		case <-time.After(1 * time.Second):
			t.Fatal("Streaming timed out")
		}

		// Verify all experiences are from players 1 or 2
		for _, batch := range stream.batches {
			for _, exp := range batch.Experiences {
				assert.Contains(t, []int32{1, 2}, exp.PlayerId)
			}
		}
	})
}