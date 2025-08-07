package experience

import (
	"math"
	"sync"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
)

// TensorPool manages reusable tensor allocations
type TensorPool struct {
	pools map[int]*sync.Pool // Keyed by tensor size
	mu    sync.RWMutex
}

// NewTensorPool creates a new tensor pool
func NewTensorPool() *TensorPool {
	return &TensorPool{
		pools: make(map[int]*sync.Pool),
	}
}

// Get retrieves a tensor from the pool or creates a new one
func (tp *TensorPool) Get(size int) []float32 {
	tp.mu.RLock()
	pool, exists := tp.pools[size]
	tp.mu.RUnlock()

	if !exists {
		tp.mu.Lock()
		// Double-check after acquiring write lock
		if pool, exists = tp.pools[size]; !exists {
			pool = &sync.Pool{
				New: func() interface{} {
					return make([]float32, size)
				},
			}
			tp.pools[size] = pool
		}
		tp.mu.Unlock()
	}

	tensor := pool.Get().([]float32)
	// Clear the tensor
	for i := range tensor {
		tensor[i] = 0
	}
	return tensor
}

// Put returns a tensor to the pool
func (tp *TensorPool) Put(tensor []float32) {
	size := len(tensor)
	tp.mu.RLock()
	pool, exists := tp.pools[size]
	tp.mu.RUnlock()

	if exists {
		pool.Put(tensor)
	}
}

// OptimizedSerializer is a performance-optimized version of the serializer
type OptimizedSerializer struct {
	tensorPool      *TensorPool
	actionMaskPool  *sync.Pool
	visibilityCache *VisibilityCache

	// Pre-computed values for common operations
	boardSizeCache map[int]*boardSizeInfo
	cacheMu        sync.RWMutex
}

// boardSizeInfo contains pre-computed values for a specific board size
type boardSizeInfo struct {
	width       int
	height      int
	tensorSize  int
	numActions  int
	channelSize int
}

// VisibilityCache caches visibility calculations per turn
type VisibilityCache struct {
	cache map[string][]bool // Key: "gameID:turn:playerID"
	mu    sync.RWMutex
	ttl   int // Number of turns to keep cache
}

// NewOptimizedSerializer creates a new optimized serializer
func NewOptimizedSerializer() *OptimizedSerializer {
	return &OptimizedSerializer{
		tensorPool: NewTensorPool(),
		actionMaskPool: &sync.Pool{
			New: func() interface{} {
				// Default size, will be resized as needed
				return make([]bool, 0)
			},
		},
		visibilityCache: &VisibilityCache{
			cache: make(map[string][]bool),
			ttl:   10, // Keep visibility for 10 turns
		},
		boardSizeCache: make(map[int]*boardSizeInfo),
	}
}

// getBoardSizeInfo returns cached board size information
func (s *OptimizedSerializer) getBoardSizeInfo(width, height int) *boardSizeInfo {
	key := width*10000 + height // Simple hash

	s.cacheMu.RLock()
	info, exists := s.boardSizeCache[key]
	s.cacheMu.RUnlock()

	if !exists {
		s.cacheMu.Lock()
		// Double-check
		if info, exists = s.boardSizeCache[key]; !exists {
			info = &boardSizeInfo{
				width:       width,
				height:      height,
				tensorSize:  NumChannels * width * height,
				numActions:  width * height * 4,
				channelSize: width * height,
			}
			s.boardSizeCache[key] = info
		}
		s.cacheMu.Unlock()
	}

	return info
}

// StateToTensor converts a game state to a tensor using pooled memory
func (s *OptimizedSerializer) StateToTensor(state *game.GameState, playerID int) []float32 {
	info := s.getBoardSizeInfo(state.Board.W, state.Board.H)

	// Get tensor from pool
	tensor := s.tensorPool.Get(info.tensorSize)

	// Pre-compute channel offsets
	channelOffsets := [NumChannels]int{}
	for i := 0; i < NumChannels; i++ {
		channelOffsets[i] = i * info.channelSize
	}

	// Get visibility info (cached if possible)
	visibility := s.getVisibility(state, playerID)

	// Process tiles in a single loop with optimized calculations
	tileIdx := 0
	for y := 0; y < info.height; y++ {
		for x := 0; x < info.width; x++ {
			tile := &state.Board.T[tileIdx]
			baseIdx := y*info.width + x

			// Check visibility
			isVisible := visibility[tileIdx]

			// Set visibility channels
			if isVisible {
				tensor[channelOffsets[ChannelVisible]+baseIdx] = 1.0
			} else {
				tensor[channelOffsets[ChannelFog]+baseIdx] = 1.0
				tileIdx++
				continue // Skip other channels if not visible
			}

			// Mountains
			if tile.IsMountain() {
				tensor[channelOffsets[ChannelMountains]+baseIdx] = 1.0
				tileIdx++
				continue
			}

			// Cities/Generals
			if tile.IsCity() || tile.IsGeneral() {
				tensor[channelOffsets[ChannelCities]+baseIdx] = 1.0
			}

			// Optimized army and territory processing
			if tile.Owner == playerID {
				// Own territory
				tensor[channelOffsets[ChannelOwnTerritory]+baseIdx] = 1.0
				if tile.Army > 0 {
					// Pre-compute normalized army value
					tensor[channelOffsets[ChannelOwnArmies]+baseIdx] = normalizeArmy(tile.Army)
				}
			} else if tile.Owner >= 0 {
				// Enemy territory
				tensor[channelOffsets[ChannelEnemyTerritory]+baseIdx] = 1.0
				if tile.Army > 0 {
					tensor[channelOffsets[ChannelEnemyArmies]+baseIdx] = normalizeArmy(tile.Army)
				}
			} else {
				// Neutral territory
				tensor[channelOffsets[ChannelNeutralTerritory]+baseIdx] = 1.0
			}

			tileIdx++
		}
	}

	return tensor
}

// getVisibility returns cached visibility or calculates it
func (s *OptimizedSerializer) getVisibility(state *game.GameState, playerID int) []bool {
	// If fog of war is disabled, return all visible
	if !state.FogOfWarEnabled {
		visibility := make([]bool, len(state.Board.T))
		for i := range visibility {
			visibility[i] = true
		}
		return visibility
	}

	// Check cache
	key := s.visibilityCacheKey(state, playerID)
	s.visibilityCache.mu.RLock()
	cached, exists := s.visibilityCache.cache[key]
	s.visibilityCache.mu.RUnlock()

	if exists {
		return cached
	}

	// Calculate visibility
	visibility := make([]bool, len(state.Board.T))
	for i, tile := range state.Board.T {
		visibility[i] = tile.IsVisibleTo(playerID)
	}

	// Cache it
	s.visibilityCache.mu.Lock()
	s.visibilityCache.cache[key] = visibility
	// TODO: Implement cache eviction based on TTL
	s.visibilityCache.mu.Unlock()

	return visibility
}

// visibilityCacheKey generates a cache key for visibility
func (s *OptimizedSerializer) visibilityCacheKey(state *game.GameState, playerID int) string {
	// Simple key format - in production, might want to use game ID
	return string(rune(state.Turn)) + ":" + string(rune(playerID))
}

// GenerateActionMask creates a boolean mask of legal actions using pooled memory
func (s *OptimizedSerializer) GenerateActionMask(state *game.GameState, playerID int) []bool {
	info := s.getBoardSizeInfo(state.Board.W, state.Board.H)

	// Get mask from pool and resize if needed
	maskInterface := s.actionMaskPool.Get()
	mask := maskInterface.([]bool)
	if cap(mask) < info.numActions {
		mask = make([]bool, info.numActions)
	} else {
		mask = mask[:info.numActions]
		// Clear the mask
		for i := range mask {
			mask[i] = false
		}
	}

	// Pre-compute direction offsets
	dirOffsets := [4]int{
		-info.width, // Up
		info.width,  // Down
		-1,          // Left
		1,           // Right
	}

	// Check each tile
	tileIdx := 0
	for y := 0; y < info.height; y++ {
		for x := 0; x < info.width; x++ {
			tile := &state.Board.T[tileIdx]

			// Can only move from tiles we own with at least 2 armies
			if tile.Owner != playerID || tile.Army < 2 {
				tileIdx++
				continue
			}

			baseActionIdx := tileIdx * 4

			// Check each direction with optimized bounds checking
			// Up
			if y > 0 {
				targetIdx := tileIdx + dirOffsets[0]
				if !state.Board.T[targetIdx].IsMountain() {
					mask[baseActionIdx] = true
				}
			}

			// Down
			if y < info.height-1 {
				targetIdx := tileIdx + dirOffsets[1]
				if !state.Board.T[targetIdx].IsMountain() {
					mask[baseActionIdx+1] = true
				}
			}

			// Left
			if x > 0 {
				targetIdx := tileIdx + dirOffsets[2]
				if !state.Board.T[targetIdx].IsMountain() {
					mask[baseActionIdx+2] = true
				}
			}

			// Right
			if x < info.width-1 {
				targetIdx := tileIdx + dirOffsets[3]
				if !state.Board.T[targetIdx].IsMountain() {
					mask[baseActionIdx+3] = true
				}
			}

			tileIdx++
		}
	}

	return mask
}

// ReturnTensor returns a tensor to the pool for reuse
func (s *OptimizedSerializer) ReturnTensor(tensor []float32) {
	s.tensorPool.Put(tensor)
}

// ReturnActionMask returns an action mask to the pool for reuse
func (s *OptimizedSerializer) ReturnActionMask(mask []bool) {
	s.actionMaskPool.Put(mask)
}

// ClearVisibilityCache clears the visibility cache
func (s *OptimizedSerializer) ClearVisibilityCache() {
	s.visibilityCache.mu.Lock()
	s.visibilityCache.cache = make(map[string][]bool)
	s.visibilityCache.mu.Unlock()
}

// normalizeArmy normalizes army value to [0, 1] range
// Optimized version without branching
func normalizeArmy(army int) float32 {
	normalized := float32(army) * (1.0 / MaxArmyValue)
	// Use math.Min to avoid branching
	return float32(math.Min(float64(normalized), 1.0))
}

// BatchStateToTensor processes multiple states in parallel
func (s *OptimizedSerializer) BatchStateToTensor(states []*game.GameState, playerID int) [][]float32 {
	results := make([][]float32, len(states))

	// Process in parallel for large batches
	if len(states) > 4 {
		var wg sync.WaitGroup
		wg.Add(len(states))

		for i, state := range states {
			go func(idx int, st *game.GameState) {
				defer wg.Done()
				results[idx] = s.StateToTensor(st, playerID)
			}(i, state)
		}

		wg.Wait()
	} else {
		// Sequential for small batches
		for i, state := range states {
			results[i] = s.StateToTensor(state, playerID)
		}
	}

	return results
}

// Implement remaining methods from original Serializer for compatibility
func (s *OptimizedSerializer) ActionToIndex(action *game.Action, boardWidth int) int {
	dirIdx := 0
	dx := action.To.X - action.From.X
	dy := action.To.Y - action.From.Y

	if dy == -1 && dx == 0 {
		dirIdx = 0 // Up
	} else if dy == 1 && dx == 0 {
		dirIdx = 1 // Down
	} else if dy == 0 && dx == -1 {
		dirIdx = 2 // Left
	} else if dy == 0 && dx == 1 {
		dirIdx = 3 // Right
	}

	return (action.From.Y*boardWidth+action.From.X)*4 + dirIdx
}

func (s *OptimizedSerializer) IndexToAction(index int, boardWidth, boardHeight int) (fromX, fromY, toX, toY int) {
	dirIdx := index % 4
	tileIdx := index / 4

	fromX = tileIdx % boardWidth
	fromY = tileIdx / boardWidth

	toX, toY = fromX, fromY
	switch dirIdx {
	case 0: // Up
		toY = fromY - 1
	case 1: // Down
		toY = fromY + 1
	case 2: // Left
		toX = fromX - 1
	case 3: // Right
		toX = fromX + 1
	}

	return fromX, fromY, toX, toY
}

func (s *OptimizedSerializer) NormalizeArmyValue(armies int) float32 {
	return normalizeArmy(armies)
}

func (s *OptimizedSerializer) ValidateAction(index int, boardWidth, boardHeight int) bool {
	maxIndex := boardWidth * boardHeight * 4
	return index >= 0 && index < maxIndex
}

func (s *OptimizedSerializer) GetTensorShape(boardWidth, boardHeight int) []int32 {
	return []int32{NumChannels, int32(boardHeight), int32(boardWidth)}
}

func (s *OptimizedSerializer) ExtractFeatures(state *game.GameState, playerID int) map[string]float32 {
	features := make(map[string]float32)

	playerArmies := float32(0)
	enemyArmies := float32(0)

	for _, tile := range state.Board.T {
		if tile.Owner == playerID {
			playerArmies += float32(tile.Army)
		} else if tile.Owner >= 0 {
			enemyArmies += float32(tile.Army)
		}
	}

	if enemyArmies > 0 {
		features["army_ratio"] = playerArmies / enemyArmies
	} else {
		features["army_ratio"] = float32(math.Inf(1))
	}

	features["turn_number"] = float32(state.Turn)
	features["total_armies"] = playerArmies

	return features
}
