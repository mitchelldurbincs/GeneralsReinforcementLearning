package gameserver

import (
	"sync"
	"time"

	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// idempotencyEntry stores a cached response with timestamp
type idempotencyEntry struct {
	response  *gamev1.SubmitActionResponse
	createdAt time.Time
}

// IdempotencyManager handles idempotent request caching
type IdempotencyManager struct {
	cache map[string]*idempotencyEntry
	mu    sync.RWMutex
}

// NewIdempotencyManager creates a new idempotency manager
func NewIdempotencyManager() *IdempotencyManager {
	return &IdempotencyManager{
		cache: make(map[string]*idempotencyEntry),
	}
}

// Check returns a cached response if the idempotency key exists
func (im *IdempotencyManager) Check(key string) *gamev1.SubmitActionResponse {
	if key == "" {
		return nil
	}
	
	im.mu.RLock()
	defer im.mu.RUnlock()
	
	entry, exists := im.cache[key]
	if !exists {
		return nil
	}
	
	// Check if entry is still valid (24 hours)
	if time.Since(entry.createdAt) > 24*time.Hour {
		return nil
	}
	
	return entry.response
}

// Store caches a response for the given idempotency key
func (im *IdempotencyManager) Store(key string, resp *gamev1.SubmitActionResponse) {
	if key == "" {
		return
	}
	
	im.mu.Lock()
	defer im.mu.Unlock()
	
	im.cache[key] = &idempotencyEntry{
		response:  resp,
		createdAt: time.Now(),
	}
	
	// Clean up old entries if cache is getting large
	if len(im.cache) > 1000 {
		im.cleanupOldEntriesLocked()
	}
}

// cleanupOldEntriesLocked removes old entries from the cache
// Must be called with mu held
func (im *IdempotencyManager) cleanupOldEntriesLocked() {
	now := time.Now()
	cutoff := now.Add(-24 * time.Hour)
	
	for key, entry := range im.cache {
		if entry.createdAt.Before(cutoff) {
			delete(im.cache, key)
		}
	}
}