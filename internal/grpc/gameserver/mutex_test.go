package gameserver

import (
	"sync"
	"testing"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// TestNoDeadlock tests that the fixed cleanup code doesn't deadlock
func TestNoDeadlock(t *testing.T) {
	gm := NewGameManager()

	// Create multiple games
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			// Create a game
			game, _ := gm.CreateGame(&gamev1.GameConfig{
				Width:      10,
				Height:     10,
				MaxPlayers: 2,
			})
			
			// Simulate some activity
			game.mu.Lock()
			game.lastActivity = time.Now().Add(-2 * abandonedGameTimeout)
			game.mu.Unlock()
		}()
	}
	
	wg.Wait()
	
	// Now run cleanup concurrently with game operations
	done := make(chan bool, 1)
	
	// Start cleanup in background
	go func() {
		for i := 0; i < 5; i++ {
			gm.cleanupGames()
			time.Sleep(10 * time.Millisecond)
		}
		done <- true
	}()
	
	// Concurrently create and access games
	go func() {
		for i := 0; i < 5; i++ {
			game, _ := gm.CreateGame(&gamev1.GameConfig{
				Width:      10,
				Height:     10,
				MaxPlayers: 2,
			})
			
			// Access game state
			game.mu.RLock()
			_ = game.currentPhaseUnlocked()
			game.mu.RUnlock()
			
			time.Sleep(10 * time.Millisecond)
		}
	}()
	
	// Wait for cleanup to complete with timeout
	select {
	case <-done:
		// Success - no deadlock
	case <-time.After(5 * time.Second):
		t.Fatal("Deadlock detected - cleanup didn't complete within 5 seconds")
	}
}

// TestConcurrentGameAccess tests that consolidated mutex doesn't cause issues
func TestConcurrentGameAccess(t *testing.T) {
	gm := NewGameManager()
	game, _ := gm.CreateGame(&gamev1.GameConfig{
		Width:      10,
		Height:     10,
		MaxPlayers: 2,
	})
	
	// Initialize the action buffer
	game.actionBuffer = make(map[int32]core.Action)
	
	var wg sync.WaitGroup
	
	// Simulate concurrent action submissions
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(playerID int32) {
			defer wg.Done()
			
			// Collect action (uses single mutex now)
			game.collectAction(playerID, nil)
			
			// Access current phase (uses RLock)
			_ = game.CurrentPhase()
		}(int32(i))
	}
	
	// Simulate concurrent reads
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			// Multiple reads should not block each other
			game.mu.RLock()
			_ = game.currentTurn
			_ = game.turnDeadline
			game.mu.RUnlock()
		}()
	}
	
	done := make(chan bool)
	go func() {
		wg.Wait()
		done <- true
	}()
	
	select {
	case <-done:
		// Success
	case <-time.After(2 * time.Second):
		t.Fatal("Concurrent access took too long - possible deadlock")
	}
}