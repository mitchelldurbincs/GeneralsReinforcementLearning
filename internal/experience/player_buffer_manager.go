package experience

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// PlayerBufferManager manages separate buffers for each player to reduce contention
type PlayerBufferManager struct {
	// Player-specific buffers
	playerBuffers sync.Map // map[int32]*PlayerBuffer

	// Configuration
	bufferCapacity int
	useLockFree    bool

	// Global statistics
	totalExperiences uint64
	activeBuffers    int32

	// Components
	logger zerolog.Logger

	// Closed state
	closed uint32
}

// PlayerBuffer wraps a buffer with player-specific metadata
type PlayerBuffer struct {
	PlayerID int32
	Buffer   ExperienceBuffer
	Created  int64 // Unix timestamp
	LastUsed int64 // Unix timestamp
}

// ExperienceBuffer interface for different buffer implementations
type ExperienceBuffer interface {
	Add(exp *experiencepb.Experience) error
	Get() (*experiencepb.Experience, error)
	GetBatch(n int) []*experiencepb.Experience
	PeekAll() []*experiencepb.Experience
	Size() int
	Close() error
}

// bufferAdapter adapts the mutex-based Buffer to ExperienceBuffer interface
type bufferAdapter struct {
	*Buffer
}

func (b *bufferAdapter) Get() (*experiencepb.Experience, error) {
	exps := b.Buffer.Get(1)
	if len(exps) == 0 {
		return nil, fmt.Errorf("buffer empty")
	}
	return exps[0], nil
}

func (b *bufferAdapter) GetBatch(n int) []*experiencepb.Experience {
	return b.Buffer.Get(n)
}

func (b *bufferAdapter) PeekAll() []*experiencepb.Experience {
	// For mutex buffer, we'll use GetAll which does remove items
	// This is a limitation of the adapter pattern
	return b.Buffer.GetAll()
}

// NewPlayerBufferManager creates a new per-player buffer manager
func NewPlayerBufferManager(bufferCapacity int, useLockFree bool, logger zerolog.Logger) *PlayerBufferManager {
	if bufferCapacity <= 0 {
		bufferCapacity = 1000
	}

	return &PlayerBufferManager{
		bufferCapacity: bufferCapacity,
		useLockFree:    useLockFree,
		logger:         logger.With().Str("component", "player_buffer_manager").Logger(),
	}
}

// GetOrCreateBuffer gets or creates a buffer for a specific player
func (m *PlayerBufferManager) GetOrCreateBuffer(playerID int32) (ExperienceBuffer, error) {
	if atomic.LoadUint32(&m.closed) == 1 {
		return nil, ErrBufferClosed
	}

	// Try to load existing buffer
	if val, ok := m.playerBuffers.Load(playerID); ok {
		pb := val.(*PlayerBuffer)
		atomic.StoreInt64(&pb.LastUsed, timeNow())
		return pb.Buffer, nil
	}

	// Create new buffer
	var buffer ExperienceBuffer
	if m.useLockFree {
		buffer = NewLockFreeBuffer(m.bufferCapacity, m.logger)
	} else {
		buffer = &bufferAdapter{NewBuffer(m.bufferCapacity, m.logger)}
	}

	pb := &PlayerBuffer{
		PlayerID: playerID,
		Buffer:   buffer,
		Created:  timeNow(),
		LastUsed: timeNow(),
	}

	// Store buffer (handle race condition)
	actual, loaded := m.playerBuffers.LoadOrStore(playerID, pb)
	if loaded {
		// Another goroutine created it first, close ours
		_ = buffer.Close()
		pb = actual.(*PlayerBuffer)
	} else {
		// We created it
		atomic.AddInt32(&m.activeBuffers, 1)
		m.logger.Debug().
			Int32("player_id", playerID).
			Bool("lock_free", m.useLockFree).
			Int("capacity", m.bufferCapacity).
			Msg("Created player buffer")
	}

	return pb.Buffer, nil
}

// AddExperience adds an experience for a specific player
func (m *PlayerBufferManager) AddExperience(playerID int32, exp *experiencepb.Experience) error {
	buffer, err := m.GetOrCreateBuffer(playerID)
	if err != nil {
		return err
	}

	if err := buffer.Add(exp); err != nil {
		return err
	}

	atomic.AddUint64(&m.totalExperiences, 1)
	return nil
}

// GetExperiences retrieves experiences for a specific player
func (m *PlayerBufferManager) GetExperiences(playerID int32, n int) ([]*experiencepb.Experience, error) {
	if val, ok := m.playerBuffers.Load(playerID); ok {
		pb := val.(*PlayerBuffer)
		// Handle special case of -1 meaning "get all"
		if n < 0 {
			n = pb.Buffer.Size()
		}
		return pb.Buffer.GetBatch(n), nil
	}
	return nil, fmt.Errorf("no buffer for player %d", playerID)
}

// GetAllExperiences retrieves all experiences from all players without removing them
func (m *PlayerBufferManager) GetAllExperiences() map[int32][]*experiencepb.Experience {
	result := make(map[int32][]*experiencepb.Experience)

	m.playerBuffers.Range(func(key, value interface{}) bool {
		playerID := key.(int32)
		pb := value.(*PlayerBuffer)

		// Peek at all experiences from this player's buffer
		experiences := pb.Buffer.PeekAll()
		if len(experiences) > 0 {
			result[playerID] = experiences
		}
		return true
	})

	return result
}

// MergeAllExperiences merges all player experiences into a single slice
func (m *PlayerBufferManager) MergeAllExperiences() []*experiencepb.Experience {
	var result []*experiencepb.Experience

	m.playerBuffers.Range(func(key, value interface{}) bool {
		pb := value.(*PlayerBuffer)
		experiences := pb.Buffer.PeekAll()
		result = append(result, experiences...)
		return true
	})

	return result
}

// RemovePlayerBuffer removes a specific player's buffer
func (m *PlayerBufferManager) RemovePlayerBuffer(playerID int32) error {
	if val, loaded := m.playerBuffers.LoadAndDelete(playerID); loaded {
		pb := val.(*PlayerBuffer)
		if err := pb.Buffer.Close(); err != nil {
			return err
		}
		atomic.AddInt32(&m.activeBuffers, -1)
		m.logger.Debug().
			Int32("player_id", playerID).
			Msg("Removed player buffer")
	}
	return nil
}

// Stats returns manager statistics
func (m *PlayerBufferManager) Stats() PlayerBufferStats {
	stats := PlayerBufferStats{
		TotalExperiences: atomic.LoadUint64(&m.totalExperiences),
		ActiveBuffers:    int(atomic.LoadInt32(&m.activeBuffers)),
		PlayerStats:      make(map[int32]PlayerBufferInfo),
	}

	m.playerBuffers.Range(func(key, value interface{}) bool {
		playerID := key.(int32)
		pb := value.(*PlayerBuffer)

		stats.PlayerStats[playerID] = PlayerBufferInfo{
			BufferSize: pb.Buffer.Size(),
			Created:    pb.Created,
			LastUsed:   atomic.LoadInt64(&pb.LastUsed),
		}
		stats.TotalBuffered += pb.Buffer.Size()
		return true
	})

	return stats
}

// Close closes all player buffers
func (m *PlayerBufferManager) Close() error {
	if !atomic.CompareAndSwapUint32(&m.closed, 0, 1) {
		return nil // Already closed
	}

	var closeErrors []error

	m.playerBuffers.Range(func(key, value interface{}) bool {
		pb := value.(*PlayerBuffer)
		if err := pb.Buffer.Close(); err != nil {
			closeErrors = append(closeErrors, err)
		}
		return true
	})

	stats := m.Stats()
	m.logger.Info().
		Uint64("total_experiences", stats.TotalExperiences).
		Int("active_buffers", stats.ActiveBuffers).
		Int("total_buffered", stats.TotalBuffered).
		Msg("Player buffer manager closed")

	if len(closeErrors) > 0 {
		return fmt.Errorf("errors closing buffers: %v", closeErrors)
	}

	return nil
}

// PlayerBufferStats contains statistics for the player buffer manager
type PlayerBufferStats struct {
	TotalExperiences uint64
	ActiveBuffers    int
	TotalBuffered    int
	PlayerStats      map[int32]PlayerBufferInfo
}

// PlayerBufferInfo contains information about a specific player's buffer
type PlayerBufferInfo struct {
	BufferSize int
	Created    int64
	LastUsed   int64
}

// timeNow returns current unix timestamp (mockable for tests)
var timeNow = func() int64 {
	return time.Now().Unix()
}

// DistributedCollector uses per-player buffers to reduce contention
type DistributedCollector struct {
	manager    *PlayerBufferManager
	gameID     string
	serializer *Serializer
	logger     zerolog.Logger
}

// NewDistributedCollector creates a collector that distributes experiences across player buffers
func NewDistributedCollector(bufferCapacity int, useLockFree bool, gameID string, logger zerolog.Logger) *DistributedCollector {
	return &DistributedCollector{
		manager:    NewPlayerBufferManager(bufferCapacity, useLockFree, logger),
		gameID:     gameID,
		serializer: NewSerializer(),
		logger:     logger.With().Str("component", "distributed_collector").Logger(),
	}
}

// OnStateTransition collects experience using per-player buffers
func (c *DistributedCollector) OnStateTransition(prevState, currState *game.GameState, actions map[int]*game.Action) {
	// Process each player's action in parallel
	var wg sync.WaitGroup

	for playerID, action := range actions {
		if action == nil {
			continue
		}

		wg.Add(1)
		go func(pid int, act *game.Action) {
			defer wg.Done()

			// Generate experience data
			expID := uuid.New().String()
			stateTensor := c.serializer.StateToTensor(prevState, pid)
			nextStateTensor := c.serializer.StateToTensor(currState, pid)
			reward := CalculateReward(prevState, currState, pid)
			actionMask := c.serializer.GenerateActionMask(prevState, pid)
			actionIndex := c.serializer.ActionToIndex(act, prevState.Board.W)
			done := currState.IsGameOver()

			// Create experience
			exp := &experiencepb.Experience{
				ExperienceId: expID,
				GameId:       c.gameID,
				PlayerId:     int32(pid),
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
					"collector_version": "distributed-1.0.0",
				},
			}

			// Add to player-specific buffer
			if err := c.manager.AddExperience(int32(pid), exp); err != nil {
				c.logger.Error().
					Err(err).
					Int("player_id", pid).
					Str("experience_id", expID).
					Msg("Failed to add experience")
			} else {
				c.logger.Debug().
					Str("experience_id", expID).
					Int("player_id", pid).
					Float32("reward", reward).
					Msg("Collected experience")
			}
		}(playerID, action)
	}

	wg.Wait()
}

// OnGameEnd handles terminal states
func (c *DistributedCollector) OnGameEnd(finalState *game.GameState) {
	stats := c.manager.Stats()
	c.logger.Info().
		Str("game_id", c.gameID).
		Uint64("total_experiences", stats.TotalExperiences).
		Int("active_buffers", stats.ActiveBuffers).
		Int("winner", finalState.GetWinner()).
		Msg("Game ended, distributed collection complete")
}

// GetExperiences returns all collected experiences
func (c *DistributedCollector) GetExperiences() []*experiencepb.Experience {
	return c.manager.MergeAllExperiences()
}

// GetPlayerExperiences returns experiences for a specific player
func (c *DistributedCollector) GetPlayerExperiences(playerID int32) ([]*experiencepb.Experience, error) {
	return c.manager.GetExperiences(playerID, -1) // -1 means get all
}

// GetExperienceCount returns the total number of experiences
func (c *DistributedCollector) GetExperienceCount() int {
	stats := c.manager.Stats()
	return stats.TotalBuffered
}

// Clear removes all experiences
func (c *DistributedCollector) Clear() {
	// Remove all player buffers
	c.manager.playerBuffers.Range(func(key, value interface{}) bool {
		playerID := key.(int32)
		_ = c.manager.RemovePlayerBuffer(playerID)
		return true
	})
}

// Close cleanly shuts down the collector
func (c *DistributedCollector) Close() error {
	return c.manager.Close()
}
