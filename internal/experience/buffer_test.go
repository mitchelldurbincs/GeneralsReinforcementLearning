package experience

import (
	"sync"
	"testing"
	"time"

	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func createTestExperience(id string, playerID int32) *experiencepb.Experience {
	return &experiencepb.Experience{
		ExperienceId: id,
		GameId:       "test-game",
		PlayerId:     playerID,
		Turn:         1,
		State: &experiencepb.TensorState{
			Shape: []int32{9, 10, 10},
			Data:  make([]float32, 900),
		},
		Action: 0,
		Reward: 0.5,
		NextState: &experiencepb.TensorState{
			Shape: []int32{9, 10, 10},
			Data:  make([]float32, 900),
		},
		Done:        false,
		ActionMask:  make([]bool, 400),
		CollectedAt: timestamppb.Now(),
	}
}

func TestBuffer_Creation(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(100, logger)
	
	assert.NotNil(t, buffer)
	assert.Equal(t, 100, buffer.Capacity())
	assert.Equal(t, 0, buffer.Size())
	assert.False(t, buffer.IsFull())
}

func TestBuffer_AddAndGet(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(10, logger)
	
	// Add experiences
	for i := 0; i < 5; i++ {
		exp := createTestExperience(string(rune('a'+i)), int32(i))
		err := buffer.Add(exp)
		assert.NoError(t, err)
	}
	
	assert.Equal(t, 5, buffer.Size())
	
	// Get experiences
	experiences := buffer.Get(3)
	assert.Len(t, experiences, 3)
	assert.Equal(t, 2, buffer.Size())
	
	// Verify order (FIFO)
	assert.Equal(t, "a", experiences[0].ExperienceId)
	assert.Equal(t, "b", experiences[1].ExperienceId)
	assert.Equal(t, "c", experiences[2].ExperienceId)
}

func TestBuffer_CircularBehavior(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(3, logger)
	
	// Fill buffer beyond capacity
	for i := 0; i < 5; i++ {
		exp := createTestExperience(string(rune('a'+i)), int32(i))
		err := buffer.Add(exp)
		assert.NoError(t, err)
	}
	
	// Buffer should contain only the last 3 experiences
	assert.Equal(t, 3, buffer.Size())
	assert.True(t, buffer.IsFull())
	
	experiences := buffer.GetAll()
	assert.Len(t, experiences, 3)
	
	// Should have c, d, e (a and b were dropped)
	assert.Equal(t, "c", experiences[0].ExperienceId)
	assert.Equal(t, "d", experiences[1].ExperienceId)
	assert.Equal(t, "e", experiences[2].ExperienceId)
	
	// Check statistics
	stats := buffer.Stats()
	assert.Equal(t, int64(5), stats.TotalAdded)
	assert.Equal(t, int64(2), stats.TotalDropped)
}

func TestBuffer_AddBatch(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(10, logger)
	
	// Create batch
	batch := make([]*experiencepb.Experience, 5)
	for i := 0; i < 5; i++ {
		batch[i] = createTestExperience(string(rune('a'+i)), int32(i))
	}
	
	err := buffer.AddBatch(batch)
	assert.NoError(t, err)
	assert.Equal(t, 5, buffer.Size())
	
	// Verify all were added
	experiences := buffer.GetAll()
	assert.Len(t, experiences, 5)
}

func TestBuffer_Sample(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(10, logger)
	
	// Add experiences
	for i := 0; i < 7; i++ {
		exp := createTestExperience(string(rune('a'+i)), int32(i))
		buffer.Add(exp)
	}
	
	// Sample fewer than available
	sample := buffer.Sample(3)
	assert.Len(t, sample, 3)
	assert.Equal(t, 7, buffer.Size()) // Sampling doesn't remove
	
	// Sample more than available
	sample = buffer.Sample(10)
	assert.Len(t, sample, 7)
}

func TestBuffer_Clear(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(10, logger)
	
	// Add some experiences
	for i := 0; i < 5; i++ {
		buffer.Add(createTestExperience(string(rune('a'+i)), int32(i)))
	}
	
	assert.Equal(t, 5, buffer.Size())
	
	// Clear
	buffer.Clear()
	assert.Equal(t, 0, buffer.Size())
	
	// Can still add after clear
	err := buffer.Add(createTestExperience("new", 0))
	assert.NoError(t, err)
	assert.Equal(t, 1, buffer.Size())
}

func TestBuffer_ConcurrentAccess(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(1000, logger)
	
	// Concurrent writers
	var wg sync.WaitGroup
	numWriters := 10
	writesPerWriter := 100
	
	for w := 0; w < numWriters; w++ {
		wg.Add(1)
		go func(writerID int) {
			defer wg.Done()
			for i := 0; i < writesPerWriter; i++ {
				exp := createTestExperience(string(rune('a'+writerID)), int32(i))
				buffer.Add(exp)
			}
		}(w)
	}
	
	// Concurrent readers
	numReaders := 5
	for r := 0; r < numReaders; r++ {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				buffer.Get(5)
				time.Sleep(time.Millisecond)
			}
		}(r)
	}
	
	wg.Wait()
	
	// Verify stats
	stats := buffer.Stats()
	assert.Equal(t, int64(numWriters*writesPerWriter), stats.TotalAdded)
}

func TestBuffer_StreamChannel(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(10, logger)
	
	// Get stream channel
	stream := buffer.StreamChannel()
	
	// Start consumer
	received := make([]*experiencepb.Experience, 0)
	done := make(chan bool)
	
	go func() {
		for exp := range stream {
			received = append(received, exp)
		}
		done <- true
	}()
	
	// Add experiences
	for i := 0; i < 3; i++ {
		buffer.Add(createTestExperience(string(rune('a'+i)), int32(i)))
	}
	
	// Give consumer time to process
	time.Sleep(10 * time.Millisecond)
	
	// Close buffer
	buffer.Close()
	
	// Wait for consumer
	<-done
	
	assert.Len(t, received, 3)
}

func TestBuffer_ClosedOperations(t *testing.T) {
	logger := zerolog.Nop()
	buffer := NewBuffer(10, logger)
	
	// Close buffer
	err := buffer.Close()
	assert.NoError(t, err)
	
	// Try operations on closed buffer
	err = buffer.Add(createTestExperience("test", 0))
	assert.Equal(t, ErrBufferClosed, err)
	
	err = buffer.AddBatch([]*experiencepb.Experience{createTestExperience("test", 0)})
	assert.Equal(t, ErrBufferClosed, err)
	
	// Close again should be no-op
	err = buffer.Close()
	assert.NoError(t, err)
}

func TestBufferManager_Creation(t *testing.T) {
	logger := zerolog.Nop()
	manager := NewBufferManager(100, logger)
	
	assert.NotNil(t, manager)
	assert.Equal(t, 100, manager.defaultCapacity)
}

func TestBufferManager_GetOrCreate(t *testing.T) {
	logger := zerolog.Nop()
	manager := NewBufferManager(50, logger)
	
	// Get new buffer
	buffer1 := manager.GetOrCreateBuffer("game1")
	assert.NotNil(t, buffer1)
	assert.Equal(t, 50, buffer1.Capacity())
	
	// Get same buffer
	buffer2 := manager.GetOrCreateBuffer("game1")
	assert.Equal(t, buffer1, buffer2) // Same instance
	
	// Get different buffer
	buffer3 := manager.GetOrCreateBuffer("game2")
	assert.NotNil(t, buffer3)
	assert.NotEqual(t, buffer1, buffer3)
}

func TestBufferManager_RemoveBuffer(t *testing.T) {
	logger := zerolog.Nop()
	manager := NewBufferManager(50, logger)
	
	// Create buffer
	buffer := manager.GetOrCreateBuffer("game1")
	buffer.Add(createTestExperience("test", 0))
	
	// Remove buffer
	err := manager.RemoveBuffer("game1")
	assert.NoError(t, err)
	
	// Verify it's gone
	_, exists := manager.GetBuffer("game1")
	assert.False(t, exists)
	
	// Remove non-existent buffer
	err = manager.RemoveBuffer("game2")
	assert.NoError(t, err) // Should not error
}

func TestBufferManager_GetAllBuffers(t *testing.T) {
	logger := zerolog.Nop()
	manager := NewBufferManager(50, logger)
	
	// Create multiple buffers
	manager.GetOrCreateBuffer("game1")
	manager.GetOrCreateBuffer("game2")
	manager.GetOrCreateBuffer("game3")
	
	all := manager.GetAllBuffers()
	assert.Len(t, all, 3)
	assert.Contains(t, all, "game1")
	assert.Contains(t, all, "game2")
	assert.Contains(t, all, "game3")
}

func TestBufferManager_CloseAll(t *testing.T) {
	logger := zerolog.Nop()
	manager := NewBufferManager(50, logger)
	
	// Create buffers
	buffer1 := manager.GetOrCreateBuffer("game1")
	buffer2 := manager.GetOrCreateBuffer("game2")
	
	// Add some data
	buffer1.Add(createTestExperience("test1", 0))
	buffer2.Add(createTestExperience("test2", 1))
	
	// Close all
	err := manager.CloseAll()
	assert.NoError(t, err)
	
	// Verify all are gone
	all := manager.GetAllBuffers()
	assert.Len(t, all, 0)
}

func TestStreamMerger(t *testing.T) {
	logger := zerolog.Nop()
	merger := NewStreamMerger(100, logger)
	
	// Create source channels
	source1 := make(chan *experiencepb.Experience, 10)
	source2 := make(chan *experiencepb.Experience, 10)
	
	merger.AddSource(source1)
	merger.AddSource(source2)
	merger.Start()
	
	// Send to sources
	go func() {
		for i := 0; i < 3; i++ {
			source1 <- createTestExperience("s1-"+string(rune('a'+i)), 0)
		}
		close(source1)
	}()
	
	go func() {
		for i := 0; i < 2; i++ {
			source2 <- createTestExperience("s2-"+string(rune('a'+i)), 1)
		}
		close(source2)
	}()
	
	// Collect from output
	output := merger.Output()
	received := make([]*experiencepb.Experience, 0)
	
	go func() {
		for exp := range output {
			received = append(received, exp)
		}
	}()
	
	// Wait a bit for processing
	time.Sleep(50 * time.Millisecond)
	
	// Close merger
	merger.Close()
	
	// Should have received all 5 experiences
	assert.Len(t, received, 5)
}

func TestTimedBatcher(t *testing.T) {
	logger := zerolog.Nop()
	input := make(chan *experiencepb.Experience, 10)
	batcher := NewTimedBatcher(input, 3, 50*time.Millisecond, logger)
	
	batcher.Start()
	
	// Send experiences
	go func() {
		for i := 0; i < 7; i++ {
			input <- createTestExperience(string(rune('a'+i)), int32(i))
			if i == 2 {
				// Pause to trigger size-based batch
				time.Sleep(10 * time.Millisecond)
			}
		}
		close(input)
	}()
	
	// Collect batches
	batches := make([][]*experiencepb.Experience, 0)
	for batch := range batcher.Output() {
		batches = append(batches, batch)
	}
	
	// Should have multiple batches
	assert.GreaterOrEqual(t, len(batches), 2)
	
	// Total experiences should be 7
	total := 0
	for _, batch := range batches {
		total += len(batch)
	}
	assert.Equal(t, 7, total)
}