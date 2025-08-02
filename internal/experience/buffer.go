package experience

import (
	"errors"
	"sync"
	"time"

	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
)

var (
	// ErrBufferFull is returned when the buffer is at capacity
	ErrBufferFull = errors.New("experience buffer is full")
	// ErrBufferClosed is returned when operations are attempted on a closed buffer
	ErrBufferClosed = errors.New("experience buffer is closed")
)

// Buffer represents a thread-safe circular buffer for experiences
type Buffer struct {
	mu         sync.RWMutex
	buffer     []*experiencepb.Experience
	capacity   int
	size       int
	head       int // Write position
	tail       int // Read position
	closed     bool
	
	// Channels for streaming
	streamChan chan *experiencepb.Experience
	closeChan  chan struct{}
	
	// Statistics
	totalAdded    int64
	totalDropped  int64
	totalStreamed int64
	
	logger zerolog.Logger
}

// NewBuffer creates a new experience buffer with the specified capacity
func NewBuffer(capacity int, logger zerolog.Logger) *Buffer {
	if capacity <= 0 {
		capacity = 10000 // Default capacity
	}
	
	return &Buffer{
		buffer:     make([]*experiencepb.Experience, capacity),
		capacity:   capacity,
		streamChan: make(chan *experiencepb.Experience, 100),
		closeChan:  make(chan struct{}),
		logger:     logger.With().Str("component", "experience_buffer").Logger(),
	}
}

// Add adds an experience to the buffer
func (b *Buffer) Add(exp *experiencepb.Experience) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	if b.closed {
		return ErrBufferClosed
	}
	
	// Check if buffer is full
	if b.size >= b.capacity {
		// Drop oldest experience (circular buffer behavior)
		b.tail = (b.tail + 1) % b.capacity
		b.totalDropped++
		b.logger.Debug().
			Int64("dropped_total", b.totalDropped).
			Msg("Buffer full, dropping oldest experience")
	} else {
		b.size++
	}
	
	// Add new experience
	b.buffer[b.head] = exp
	b.head = (b.head + 1) % b.capacity
	b.totalAdded++
	
	// Try to stream experience (non-blocking)
	select {
	case b.streamChan <- exp:
		b.totalStreamed++
	default:
		// Stream channel full, continue without blocking
		b.logger.Debug().Msg("Stream channel full, experience not streamed")
	}
	
	return nil
}

// AddBatch adds multiple experiences to the buffer
func (b *Buffer) AddBatch(experiences []*experiencepb.Experience) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	if b.closed {
		return ErrBufferClosed
	}
	
	for _, exp := range experiences {
		// Inline add logic to avoid lock contention
		if b.size >= b.capacity {
			b.tail = (b.tail + 1) % b.capacity
			b.totalDropped++
		} else {
			b.size++
		}
		
		b.buffer[b.head] = exp
		b.head = (b.head + 1) % b.capacity
		b.totalAdded++
		
		// Try to stream
		select {
		case b.streamChan <- exp:
			b.totalStreamed++
		default:
			// Continue without blocking
		}
	}
	
	if len(experiences) > 0 {
		b.logger.Debug().
			Int("batch_size", len(experiences)).
			Int64("total_added", b.totalAdded).
			Msg("Added batch of experiences")
	}
	
	return nil
}

// Get retrieves up to n experiences from the buffer
func (b *Buffer) Get(n int) []*experiencepb.Experience {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	if n > b.size {
		n = b.size
	}
	
	result := make([]*experiencepb.Experience, n)
	for i := 0; i < n; i++ {
		result[i] = b.buffer[b.tail]
		b.tail = (b.tail + 1) % b.capacity
		b.size--
	}
	
	return result
}

// GetAll retrieves all experiences from the buffer
func (b *Buffer) GetAll() []*experiencepb.Experience {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	if b.size == 0 {
		return []*experiencepb.Experience{}
	}
	
	result := make([]*experiencepb.Experience, b.size)
	for i := 0; i < b.size; i++ {
		result[i] = b.buffer[b.tail]
		b.tail = (b.tail + 1) % b.capacity
	}
	b.size = 0
	
	return result
}

// Sample randomly samples n experiences from the buffer
func (b *Buffer) Sample(n int) []*experiencepb.Experience {
	b.mu.RLock()
	defer b.mu.RUnlock()
	
	if n > b.size {
		n = b.size
	}
	
	// For now, just return the latest n experiences
	// TODO: Implement true random sampling
	result := make([]*experiencepb.Experience, n)
	for i := 0; i < n; i++ {
		idx := (b.tail + i) % b.capacity
		result[i] = b.buffer[idx]
	}
	
	return result
}

// GetLatest returns the n most recent experiences from the buffer
func (b *Buffer) GetLatest(n int) []*experiencepb.Experience {
	b.mu.RLock()
	defer b.mu.RUnlock()
	
	if n > b.size {
		n = b.size
	}
	
	result := make([]*experiencepb.Experience, n)
	// Start from head and go backwards to get latest experiences
	for i := 0; i < n; i++ {
		// Calculate index going backwards from head
		idx := (b.head - n + i + b.capacity) % b.capacity
		result[i] = b.buffer[idx]
	}
	
	return result
}

// StreamChannel returns a channel for streaming experiences
func (b *Buffer) StreamChannel() <-chan *experiencepb.Experience {
	return b.streamChan
}

// Size returns the current number of experiences in the buffer
func (b *Buffer) Size() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.size
}

// Capacity returns the maximum capacity of the buffer
func (b *Buffer) Capacity() int {
	return b.capacity
}

// IsFull returns true if the buffer is at capacity
func (b *Buffer) IsFull() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.size >= b.capacity
}

// Clear removes all experiences from the buffer
func (b *Buffer) Clear() {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	b.size = 0
	b.head = 0
	b.tail = 0
	b.buffer = make([]*experiencepb.Experience, b.capacity)
	
	b.logger.Debug().Msg("Buffer cleared")
}

// Close closes the buffer and releases resources
func (b *Buffer) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	
	if b.closed {
		return nil
	}
	
	b.closed = true
	close(b.closeChan)
	close(b.streamChan)
	
	b.logger.Info().
		Int64("total_added", b.totalAdded).
		Int64("total_dropped", b.totalDropped).
		Int64("total_streamed", b.totalStreamed).
		Msg("Buffer closed")
	
	return nil
}

// Stats returns buffer statistics
func (b *Buffer) Stats() BufferStats {
	b.mu.RLock()
	defer b.mu.RUnlock()
	
	return BufferStats{
		CurrentSize:   b.size,
		Capacity:      b.capacity,
		TotalAdded:    b.totalAdded,
		TotalDropped:  b.totalDropped,
		TotalStreamed: b.totalStreamed,
		UtilizationPct: float64(b.size) / float64(b.capacity) * 100,
	}
}

// BufferStats contains buffer statistics
type BufferStats struct {
	CurrentSize    int
	Capacity       int
	TotalAdded     int64
	TotalDropped   int64
	TotalStreamed  int64
	UtilizationPct float64
}

// BufferManager manages multiple buffers for different games or players
type BufferManager struct {
	mu      sync.RWMutex
	buffers map[string]*Buffer
	logger  zerolog.Logger
	
	// Default configuration
	defaultCapacity int
}

// NewBufferManager creates a new buffer manager
func NewBufferManager(defaultCapacity int, logger zerolog.Logger) *BufferManager {
	return &BufferManager{
		buffers:         make(map[string]*Buffer),
		defaultCapacity: defaultCapacity,
		logger:          logger.With().Str("component", "buffer_manager").Logger(),
	}
}

// GetOrCreateBuffer gets an existing buffer or creates a new one
func (m *BufferManager) GetOrCreateBuffer(key string) *Buffer {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if buffer, exists := m.buffers[key]; exists {
		return buffer
	}
	
	// Create new buffer
	buffer := NewBuffer(m.defaultCapacity, m.logger)
	m.buffers[key] = buffer
	
	m.logger.Debug().
		Str("key", key).
		Int("capacity", m.defaultCapacity).
		Msg("Created new buffer")
	
	return buffer
}

// GetBuffer retrieves a buffer by key
func (m *BufferManager) GetBuffer(key string) (*Buffer, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	buffer, exists := m.buffers[key]
	return buffer, exists
}

// RemoveBuffer removes and closes a buffer
func (m *BufferManager) RemoveBuffer(key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if buffer, exists := m.buffers[key]; exists {
		if err := buffer.Close(); err != nil {
			return err
		}
		delete(m.buffers, key)
		
		m.logger.Debug().
			Str("key", key).
			Msg("Removed buffer")
	}
	
	return nil
}

// GetAllBuffers returns all managed buffers
func (m *BufferManager) GetAllBuffers() map[string]*Buffer {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	// Return a copy to avoid concurrent modification
	result := make(map[string]*Buffer, len(m.buffers))
	for k, v := range m.buffers {
		result[k] = v
	}
	return result
}

// CloseAll closes all managed buffers
func (m *BufferManager) CloseAll() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	for key, buffer := range m.buffers {
		if err := buffer.Close(); err != nil {
			m.logger.Error().
				Err(err).
				Str("key", key).
				Msg("Failed to close buffer")
		}
	}
	
	m.buffers = make(map[string]*Buffer)
	return nil
}

// StreamMerger merges multiple experience streams into one
type StreamMerger struct {
	sources    []<-chan *experiencepb.Experience
	output     chan *experiencepb.Experience
	closeChan  chan struct{}
	wg         sync.WaitGroup
	logger     zerolog.Logger
}

// NewStreamMerger creates a new stream merger
func NewStreamMerger(bufferSize int, logger zerolog.Logger) *StreamMerger {
	return &StreamMerger{
		output:    make(chan *experiencepb.Experience, bufferSize),
		closeChan: make(chan struct{}),
		logger:    logger.With().Str("component", "stream_merger").Logger(),
	}
}

// AddSource adds a new source stream to merge
func (m *StreamMerger) AddSource(source <-chan *experiencepb.Experience) {
	m.sources = append(m.sources, source)
}

// Start begins merging streams
func (m *StreamMerger) Start() {
	for i, source := range m.sources {
		m.wg.Add(1)
		go m.mergeSource(i, source)
	}
}

// mergeSource merges a single source into the output
func (m *StreamMerger) mergeSource(sourceID int, source <-chan *experiencepb.Experience) {
	defer m.wg.Done()
	
	for {
		select {
		case exp, ok := <-source:
			if !ok {
				m.logger.Debug().
					Int("source_id", sourceID).
					Msg("Source channel closed")
				return
			}
			
			select {
			case m.output <- exp:
				// Successfully forwarded
			case <-m.closeChan:
				return
			}
			
		case <-m.closeChan:
			return
		}
	}
}

// Output returns the merged output channel
func (m *StreamMerger) Output() <-chan *experiencepb.Experience {
	return m.output
}

// Close stops the merger and closes output channel
func (m *StreamMerger) Close() {
	close(m.closeChan)
	m.wg.Wait()
	close(m.output)
}

// TimedBatcher batches experiences based on time or size
type TimedBatcher struct {
	input      <-chan *experiencepb.Experience
	output     chan []*experiencepb.Experience
	batchSize  int
	timeout    time.Duration
	closeChan  chan struct{}
	logger     zerolog.Logger
}

// NewTimedBatcher creates a new timed batcher
func NewTimedBatcher(input <-chan *experiencepb.Experience, batchSize int, timeout time.Duration, logger zerolog.Logger) *TimedBatcher {
	return &TimedBatcher{
		input:     input,
		output:    make(chan []*experiencepb.Experience, 10),
		batchSize: batchSize,
		timeout:   timeout,
		closeChan: make(chan struct{}),
		logger:    logger.With().Str("component", "timed_batcher").Logger(),
	}
}

// Start begins batching
func (b *TimedBatcher) Start() {
	go b.run()
}

// run is the main batching loop
func (b *TimedBatcher) run() {
	batch := make([]*experiencepb.Experience, 0, b.batchSize)
	timer := time.NewTimer(b.timeout)
	defer timer.Stop()
	
	for {
		select {
		case exp, ok := <-b.input:
			if !ok {
				// Input closed, flush remaining batch
				if len(batch) > 0 {
					b.output <- batch
				}
				close(b.output)
				return
			}
			
			batch = append(batch, exp)
			
			if len(batch) >= b.batchSize {
				// Batch full, send it
				b.output <- batch
				batch = make([]*experiencepb.Experience, 0, b.batchSize)
				timer.Reset(b.timeout)
			}
			
		case <-timer.C:
			// Timeout reached, send partial batch
			if len(batch) > 0 {
				b.output <- batch
				batch = make([]*experiencepb.Experience, 0, b.batchSize)
			}
			timer.Reset(b.timeout)
			
		case <-b.closeChan:
			// Flush remaining batch
			if len(batch) > 0 {
				b.output <- batch
			}
			close(b.output)
			return
		}
	}
}

// Output returns the batched output channel
func (b *TimedBatcher) Output() <-chan []*experiencepb.Experience {
	return b.output
}

// Close stops the batcher
func (b *TimedBatcher) Close() {
	close(b.closeChan)
}