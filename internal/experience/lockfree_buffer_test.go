package experience

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLockFreeBuffer_Creation(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))

	// Test with valid capacity
	buf := NewLockFreeBuffer(100, logger)
	assert.NotNil(t, buf)
	assert.Equal(t, 128, buf.Capacity()) // Rounded up to power of 2
	assert.Equal(t, 0, buf.Size())
	assert.False(t, buf.IsFull())

	// Test with zero capacity (should use default)
	buf2 := NewLockFreeBuffer(0, logger)
	assert.Equal(t, 8192, buf2.Capacity())

	// Test power of 2 capacity
	buf3 := NewLockFreeBuffer(256, logger)
	assert.Equal(t, 256, buf3.Capacity())
}

func TestLockFreeBuffer_AddAndGet(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	buf := NewLockFreeBuffer(16, logger)

	// Add experiences
	exp1 := &experiencepb.Experience{ExperienceId: "1", Reward: 1.0}
	exp2 := &experiencepb.Experience{ExperienceId: "2", Reward: 2.0}

	err := buf.Add(exp1)
	require.NoError(t, err)
	assert.Equal(t, 1, buf.Size())

	err = buf.Add(exp2)
	require.NoError(t, err)
	assert.Equal(t, 2, buf.Size())

	// Get experiences
	got1, err := buf.Get()
	require.NoError(t, err)
	assert.Equal(t, "1", got1.ExperienceId)
	assert.Equal(t, 1, buf.Size())

	got2, err := buf.Get()
	require.NoError(t, err)
	assert.Equal(t, "2", got2.ExperienceId)
	assert.Equal(t, 0, buf.Size())

	// Try to get from empty buffer
	_, err = buf.Get()
	assert.Error(t, err)
}

func TestLockFreeBuffer_Overflow(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	buf := NewLockFreeBuffer(4, logger) // Small buffer

	// Fill buffer
	for i := 0; i < 4; i++ {
		exp := &experiencepb.Experience{ExperienceId: string(rune('A' + i))}
		err := buf.Add(exp)
		require.NoError(t, err)
	}

	assert.True(t, buf.IsFull())
	stats := buf.Stats()
	assert.Equal(t, uint64(4), stats.TotalAdded)
	assert.Equal(t, uint64(0), stats.TotalDropped)

	// Add one more (should drop oldest)
	exp := &experiencepb.Experience{ExperienceId: "E"}
	err := buf.Add(exp)
	require.NoError(t, err)

	stats = buf.Stats()
	assert.Equal(t, uint64(5), stats.TotalAdded)
	assert.Equal(t, uint64(1), stats.TotalDropped)
	assert.Equal(t, 4, stats.CurrentSize)
}

func TestLockFreeBuffer_GetBatch(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	buf := NewLockFreeBuffer(16, logger)

	// Add 10 experiences
	for i := 0; i < 10; i++ {
		exp := &experiencepb.Experience{ExperienceId: string(rune('0' + i))}
		err := buf.Add(exp)
		require.NoError(t, err)
	}

	// Get batch of 5
	batch := buf.GetBatch(5)
	assert.Len(t, batch, 5)
	assert.Equal(t, 5, buf.Size())

	// Get batch larger than remaining
	batch = buf.GetBatch(10)
	assert.Len(t, batch, 5)
	assert.Equal(t, 0, buf.Size())
}

func TestLockFreeBuffer_ConcurrentAccess(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	buf := NewLockFreeBuffer(1024, logger)

	const numProducers = 10
	const numConsumers = 5
	const itemsPerProducer = 100

	var wg sync.WaitGroup

	// Producers
	for p := 0; p < numProducers; p++ {
		wg.Add(1)
		go func(producerID int) {
			defer wg.Done()
			for i := 0; i < itemsPerProducer; i++ {
				exp := &experiencepb.Experience{
					ExperienceId: string(rune('A'+producerID)) + string(rune('0'+i)),
					PlayerId:     int32(producerID),
				}
				err := buf.Add(exp)
				if err != nil {
					t.Errorf("Producer %d failed to add: %v", producerID, err)
				}
			}
		}(p)
	}

	// Let producers get ahead
	time.Sleep(10 * time.Millisecond)

	// Consumers
	consumed := make([]int32, numConsumers)
	for c := 0; c < numConsumers; c++ {
		wg.Add(1)
		go func(consumerID int) {
			defer wg.Done()
			count := int32(0)
			for {
				_, err := buf.Get()
				if err != nil {
					// Buffer empty, retry
					if atomic.LoadInt64(&buf.size) == 0 && count > 0 {
						break
					}
					time.Sleep(time.Microsecond)
					continue
				}
				count++
				atomic.AddInt32(&consumed[consumerID], 1)
				if count >= int32(itemsPerProducer*numProducers/numConsumers) {
					break
				}
			}
		}(c)
	}

	wg.Wait()

	// Verify all items were produced
	stats := buf.Stats()
	assert.Equal(t, uint64(numProducers*itemsPerProducer), stats.TotalAdded)

	// Verify consumption
	totalConsumed := int32(0)
	for _, count := range consumed {
		totalConsumed += count
	}
	// Should have consumed most items (some may be left in buffer)
	assert.Greater(t, int(totalConsumed), numProducers*itemsPerProducer-buf.capacity)
}

func TestLockFreeBuffer_Close(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	buf := NewLockFreeBuffer(16, logger)

	// Add some experiences
	exp1 := &experiencepb.Experience{ExperienceId: "1"}
	err := buf.Add(exp1)
	require.NoError(t, err)

	// Close buffer
	err = buf.Close()
	require.NoError(t, err)

	// Operations should fail after close
	exp2 := &experiencepb.Experience{ExperienceId: "2"}
	err = buf.Add(exp2)
	assert.Equal(t, ErrBufferClosed, err)

	_, err = buf.Get()
	assert.Equal(t, ErrBufferClosed, err)

	// Close again should be idempotent
	err = buf.Close()
	assert.NoError(t, err)
}

func BenchmarkLockFreeBuffer_Add(b *testing.B) {
	logger := zerolog.Nop()
	buf := NewLockFreeBuffer(8192, logger)
	exp := &experiencepb.Experience{ExperienceId: "bench", Reward: 1.0}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = buf.Add(exp)
		}
	})

	b.StopTimer()
	stats := buf.Stats()
	b.ReportMetric(float64(stats.TotalAdded)/b.Elapsed().Seconds(), "adds/sec")
}

func BenchmarkLockFreeBuffer_Get(b *testing.B) {
	logger := zerolog.Nop()
	buf := NewLockFreeBuffer(8192, logger)

	// Pre-fill buffer
	exp := &experiencepb.Experience{ExperienceId: "bench", Reward: 1.0}
	for i := 0; i < buf.capacity/2; i++ {
		_ = buf.Add(exp)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := buf.Get(); err != nil {
				// Re-add to keep buffer from emptying
				_ = buf.Add(exp)
			}
		}
	})

	b.StopTimer()
	stats := buf.Stats()
	b.ReportMetric(float64(stats.TotalRetrieved)/b.Elapsed().Seconds(), "gets/sec")
}

func BenchmarkLockFreeBuffer_Mixed(b *testing.B) {
	logger := zerolog.Nop()
	buf := NewLockFreeBuffer(8192, logger)
	exp := &experiencepb.Experience{ExperienceId: "bench", Reward: 1.0}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%2 == 0 {
				_ = buf.Add(exp)
			} else {
				_, _ = buf.Get()
			}
			i++
		}
	})

	b.StopTimer()
	stats := buf.Stats()
	total := stats.TotalAdded + stats.TotalRetrieved
	b.ReportMetric(float64(total)/b.Elapsed().Seconds(), "ops/sec")
}

// Benchmark comparison with mutex-based buffer
func BenchmarkMutexBuffer_Add(b *testing.B) {
	logger := zerolog.Nop()
	buf := NewBuffer(8192, logger)
	exp := &experiencepb.Experience{ExperienceId: "bench", Reward: 1.0}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = buf.Add(exp)
		}
	})

	b.StopTimer()
	stats := buf.Stats()
	b.ReportMetric(float64(stats.TotalAdded)/b.Elapsed().Seconds(), "adds/sec")
}

func BenchmarkMutexBuffer_Get(b *testing.B) {
	logger := zerolog.Nop()
	buf := NewBuffer(8192, logger)

	// Pre-fill buffer
	exp := &experiencepb.Experience{ExperienceId: "bench", Reward: 1.0}
	for i := 0; i < buf.capacity/2; i++ {
		_ = buf.Add(exp)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			exps := buf.Get(1)
			if len(exps) == 0 {
				// Re-add to keep buffer from emptying
				_ = buf.Add(exp)
			}
		}
	})
}

func BenchmarkMutexBuffer_Mixed(b *testing.B) {
	logger := zerolog.Nop()
	buf := NewBuffer(8192, logger)
	exp := &experiencepb.Experience{ExperienceId: "bench", Reward: 1.0}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%2 == 0 {
				_ = buf.Add(exp)
			} else {
				_ = buf.Get(1)
			}
			i++
		}
	})

	b.StopTimer()
	stats := buf.Stats()
	total := stats.TotalAdded + stats.TotalDropped
	b.ReportMetric(float64(total)/b.Elapsed().Seconds(), "ops/sec")
}
