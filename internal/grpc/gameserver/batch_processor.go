package gameserver

import (
	"context"
	"sync"
	"time"

	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
)

// BatchProcessor collects experiences into batches for efficient streaming
type BatchProcessor struct {
	batchSize    int32
	batchTimeout time.Duration
	
	// Current batch
	mu           sync.Mutex
	currentBatch []*experiencepb.Experience
	lastFlush    time.Time
	
	// Input channel
	inputCh      chan *experiencepb.Experience
	
	// Output channel
	outputCh     chan []*experiencepb.Experience
	
	// Control
	flushCh      chan struct{}
	done         chan struct{}
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor(batchSize int32, batchTimeout time.Duration) *BatchProcessor {
	if batchSize <= 0 {
		batchSize = 32
	}
	if batchTimeout <= 0 {
		batchTimeout = 100 * time.Millisecond
	}
	
	return &BatchProcessor{
		batchSize:    batchSize,
		batchTimeout: batchTimeout,
		currentBatch: make([]*experiencepb.Experience, 0, batchSize),
		inputCh:      make(chan *experiencepb.Experience, 1000), // Increased buffer size
		outputCh:     make(chan []*experiencepb.Experience, 100), // Increased buffer size
		flushCh:      make(chan struct{}, 1),
		done:         make(chan struct{}),
		lastFlush:    time.Now(),
	}
}

// Start begins processing experiences into batches
func (bp *BatchProcessor) Start(ctx context.Context) <-chan []*experiencepb.Experience {
	go bp.run(ctx)
	return bp.outputCh
}

// Add adds an experience to be batched
// Returns false if the experience couldn't be added (backpressure)
func (bp *BatchProcessor) Add(exp *experiencepb.Experience) bool {
	select {
	case bp.inputCh <- exp:
		return true
	default:
		// Channel full, backpressure
		return false
	}
}

// Flush forces the current batch to be sent
func (bp *BatchProcessor) Flush() {
	select {
	case bp.flushCh <- struct{}{}:
	default:
		// Flush already pending
	}
}

// run is the main processing loop
func (bp *BatchProcessor) run(ctx context.Context) {
	defer close(bp.outputCh)
	defer close(bp.done)
	
	ticker := time.NewTicker(bp.batchTimeout)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			// Context cancelled, flush remaining batch
			bp.flushBatch()
			return
			
		case exp := <-bp.inputCh:
			// Add to current batch
			bp.mu.Lock()
			bp.currentBatch = append(bp.currentBatch, exp)
			shouldFlush := int32(len(bp.currentBatch)) >= bp.batchSize
			bp.mu.Unlock()
			
			if shouldFlush {
				bp.flushBatch()
				ticker.Reset(bp.batchTimeout)
			}
			
		case <-ticker.C:
			// Timeout, flush current batch if not empty
			bp.mu.Lock()
			hasData := len(bp.currentBatch) > 0
			bp.mu.Unlock()
			
			if hasData {
				bp.flushBatch()
			}
			
		case <-bp.flushCh:
			// Manual flush requested
			bp.flushBatch()
			ticker.Reset(bp.batchTimeout)
		}
	}
}

// flushBatch sends the current batch to the output channel
func (bp *BatchProcessor) flushBatch() {
	bp.mu.Lock()
	
	if len(bp.currentBatch) == 0 {
		bp.mu.Unlock()
		return
	}
	
	// Create a copy of the batch
	batch := make([]*experiencepb.Experience, len(bp.currentBatch))
	copy(batch, bp.currentBatch)
	
	// Reset current batch
	bp.currentBatch = bp.currentBatch[:0]
	bp.lastFlush = time.Now()
	
	bp.mu.Unlock()
	
	// Send batch to output channel (non-blocking with timeout)
	select {
	case bp.outputCh <- batch:
		// Successfully sent
	case <-time.After(time.Second):
		// Timeout, drop batch to prevent blocking
	}
}

// Stats returns batch processor statistics
func (bp *BatchProcessor) Stats() map[string]interface{} {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	
	return map[string]interface{}{
		"batch_size":       bp.batchSize,
		"batch_timeout":    bp.batchTimeout.String(),
		"current_batch":    len(bp.currentBatch),
		"input_queue":      len(bp.inputCh),
		"output_queue":     len(bp.outputCh),
		"last_flush":       bp.lastFlush,
	}
}