package experience

import (
	"errors"
	"sync/atomic"
	"unsafe"

	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
)

// LockFreeBuffer implements a lock-free ring buffer for experiences using atomic operations
type LockFreeBuffer struct {
	// Ring buffer storage
	buffer []unsafe.Pointer // Store pointers to experiences
	mask   uint64           // Capacity - 1 (for fast modulo)

	// Atomic counters
	writePos uint64 // Next write position
	readPos  uint64 // Next read position
	size     int64  // Current number of elements

	// Statistics (atomic)
	totalAdded     uint64
	totalDropped   uint64
	totalRetrieved uint64

	// Configuration
	capacity int
	logger   zerolog.Logger

	// Closed state
	closed uint32 // 0 = open, 1 = closed
}

// NewLockFreeBuffer creates a new lock-free experience buffer
// Capacity must be a power of 2 for optimal performance
func NewLockFreeBuffer(capacity int, logger zerolog.Logger) *LockFreeBuffer {
	// Ensure capacity is power of 2
	if capacity <= 0 {
		capacity = 8192
	}
	capacity = nextPowerOf2(capacity)

	buffer := make([]unsafe.Pointer, capacity)

	return &LockFreeBuffer{
		buffer:   buffer,
		mask:     uint64(capacity - 1),
		capacity: capacity,
		logger:   logger.With().Str("component", "lockfree_buffer").Logger(),
	}
}

// Add adds an experience to the buffer without locks
func (b *LockFreeBuffer) Add(exp *experiencepb.Experience) error {
	if atomic.LoadUint32(&b.closed) == 1 {
		return ErrBufferClosed
	}

	// Get next write position
	writePos := atomic.AddUint64(&b.writePos, 1) - 1
	idx := writePos & b.mask

	// Check if we're overwriting
	currentSize := atomic.LoadInt64(&b.size)
	if currentSize >= int64(b.capacity) {
		// Buffer full, we're overwriting oldest
		atomic.AddUint64(&b.totalDropped, 1)
		atomic.AddUint64(&b.readPos, 1)
	} else {
		// Increment size
		atomic.AddInt64(&b.size, 1)
	}

	// Store experience
	atomic.StorePointer(&b.buffer[idx], unsafe.Pointer(exp))
	atomic.AddUint64(&b.totalAdded, 1)

	return nil
}

// Get retrieves an experience from the buffer without locks
func (b *LockFreeBuffer) Get() (*experiencepb.Experience, error) {
	if atomic.LoadUint32(&b.closed) == 1 {
		return nil, ErrBufferClosed
	}

	// Check if buffer is empty
	if atomic.LoadInt64(&b.size) <= 0 {
		return nil, errors.New("buffer empty")
	}

	// Try to claim a read position
	for {
		readPos := atomic.LoadUint64(&b.readPos)
		writePos := atomic.LoadUint64(&b.writePos)

		// Check if empty
		if readPos >= writePos {
			return nil, errors.New("buffer empty")
		}

		// Try to claim this position
		if atomic.CompareAndSwapUint64(&b.readPos, readPos, readPos+1) {
			// Successfully claimed position
			idx := readPos & b.mask

			// Load experience
			ptr := atomic.LoadPointer(&b.buffer[idx])
			if ptr == nil {
				// Race condition, retry
				continue
			}

			exp := (*experiencepb.Experience)(ptr)

			// Clear slot
			atomic.StorePointer(&b.buffer[idx], nil)

			// Decrement size
			atomic.AddInt64(&b.size, -1)
			atomic.AddUint64(&b.totalRetrieved, 1)

			return exp, nil
		}
		// CAS failed, retry
	}
}

// GetBatch retrieves multiple experiences atomically
func (b *LockFreeBuffer) GetBatch(n int) []*experiencepb.Experience {
	if atomic.LoadUint32(&b.closed) == 1 {
		return nil
	}

	result := make([]*experiencepb.Experience, 0, n)

	for i := 0; i < n; i++ {
		exp, err := b.Get()
		if err != nil {
			break
		}
		result = append(result, exp)
	}

	return result
}

// PeekAll returns all experiences without removing them
func (b *LockFreeBuffer) PeekAll() []*experiencepb.Experience {
	if atomic.LoadUint32(&b.closed) == 1 {
		return nil
	}

	size := atomic.LoadInt64(&b.size)
	if size <= 0 {
		return []*experiencepb.Experience{}
	}

	result := make([]*experiencepb.Experience, 0, size)
	readPos := atomic.LoadUint64(&b.readPos)

	for i := int64(0); i < size; i++ {
		idx := (readPos + uint64(i)) & b.mask
		ptr := atomic.LoadPointer(&b.buffer[idx])
		if ptr != nil {
			exp := (*experiencepb.Experience)(ptr)
			result = append(result, exp)
		}
	}

	return result
}

// Size returns the current number of experiences in the buffer
func (b *LockFreeBuffer) Size() int {
	return int(atomic.LoadInt64(&b.size))
}

// Capacity returns the buffer capacity
func (b *LockFreeBuffer) Capacity() int {
	return b.capacity
}

// IsFull returns true if the buffer is at capacity
func (b *LockFreeBuffer) IsFull() bool {
	return atomic.LoadInt64(&b.size) >= int64(b.capacity)
}

// Stats returns buffer statistics
func (b *LockFreeBuffer) Stats() LockFreeBufferStats {
	size := atomic.LoadInt64(&b.size)
	return LockFreeBufferStats{
		CurrentSize:    int(size),
		Capacity:       b.capacity,
		TotalAdded:     atomic.LoadUint64(&b.totalAdded),
		TotalDropped:   atomic.LoadUint64(&b.totalDropped),
		TotalRetrieved: atomic.LoadUint64(&b.totalRetrieved),
		UtilizationPct: float64(size) / float64(b.capacity) * 100,
	}
}

// Close closes the buffer
func (b *LockFreeBuffer) Close() error {
	if !atomic.CompareAndSwapUint32(&b.closed, 0, 1) {
		return nil // Already closed
	}

	stats := b.Stats()
	b.logger.Info().
		Uint64("total_added", stats.TotalAdded).
		Uint64("total_dropped", stats.TotalDropped).
		Uint64("total_retrieved", stats.TotalRetrieved).
		Msg("Lock-free buffer closed")

	return nil
}

// LockFreeBufferStats contains statistics for the lock-free buffer
type LockFreeBufferStats struct {
	CurrentSize    int
	Capacity       int
	TotalAdded     uint64
	TotalDropped   uint64
	TotalRetrieved uint64
	UtilizationPct float64
}

// Helper function to get next power of 2
func nextPowerOf2(n int) int {
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	n++
	return n
}

// MultiProducerBuffer wraps LockFreeBuffer with per-producer batching
type MultiProducerBuffer struct {
	core      *LockFreeBuffer
	producers map[string]*ProducerBatch
	logger    zerolog.Logger
}

// ProducerBatch holds a batch of experiences for a single producer
type ProducerBatch struct {
	// batch and capacity are kept for future batch operations
	//nolint:unused // Will be used for batch operations
	batch []*experiencepb.Experience
	//nolint:unused // Will be used for batch size optimization
	capacity int
}

// NewMultiProducerBuffer creates a buffer optimized for multiple producers
func NewMultiProducerBuffer(capacity int, producerBatchSize int, logger zerolog.Logger) *MultiProducerBuffer {
	return &MultiProducerBuffer{
		core:      NewLockFreeBuffer(capacity, logger),
		producers: make(map[string]*ProducerBatch),
		logger:    logger.With().Str("component", "multi_producer_buffer").Logger(),
	}
}

// AddForProducer adds an experience for a specific producer
func (m *MultiProducerBuffer) AddForProducer(producerID string, exp *experiencepb.Experience) error {
	// This is a simplified version - in production you'd want thread-local storage
	// or a more sophisticated producer identification mechanism
	return m.core.Add(exp)
}

// GetCore returns the underlying lock-free buffer
func (m *MultiProducerBuffer) GetCore() *LockFreeBuffer {
	return m.core
}
