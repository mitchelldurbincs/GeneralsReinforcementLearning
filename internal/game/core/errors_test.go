package core

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWrapActionError(t *testing.T) {
	tests := []struct {
		name     string
		action   Action
		err      error
		expected string
		isNil    bool
	}{
		{
			name:     "nil error returns nil",
			action:   &MoveAction{PlayerID: 1, FromX: 0, FromY: 0, ToX: 1, ToY: 0},
			err:      nil,
			expected: "",
			isNil:    true,
		},
		{
			name:     "move action with coordinates",
			action:   &MoveAction{PlayerID: 1, FromX: 5, FromY: 3, ToX: 5, ToY: 4},
			err:      ErrInvalidCoordinates,
			expected: "player 1: move from (5,3) to (5,4): invalid coordinates",
		},
		{
			name:     "move action with not owned error",
			action:   &MoveAction{PlayerID: 2, FromX: 10, FromY: 10, ToX: 11, ToY: 10},
			err:      ErrNotOwned,
			expected: "player 2: move from (10,10) to (11,10): tile not owned by player",
		},
		{
			name:     "generic action fallback",
			action:   nil,
			err:      ErrGameOver,
			expected: "player action: game is over",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wrapped := WrapActionError(tt.action, tt.err)
			if tt.isNil {
				assert.Nil(t, wrapped)
			} else {
				require.NotNil(t, wrapped)
				assert.Equal(t, tt.expected, wrapped.Error())
				// Verify error unwrapping works
				assert.True(t, errors.Is(wrapped, tt.err))
			}
		})
	}
}

func TestWrapGameStateError(t *testing.T) {
	tests := []struct {
		name     string
		turn     int
		phase    string
		err      error
		expected string
		isNil    bool
	}{
		{
			name:     "nil error returns nil",
			turn:     50,
			phase:    "action",
			err:      nil,
			expected: "",
			isNil:    true,
		},
		{
			name:     "action phase error",
			turn:     100,
			phase:    "action",
			err:      ErrGameOver,
			expected: "game turn 100 [action]: game is over",
		},
		{
			name:     "production phase error",
			turn:     25,
			phase:    "production",
			err:      fmt.Errorf("failed to apply production"),
			expected: "game turn 25 [production]: failed to apply production",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wrapped := WrapGameStateError(tt.turn, tt.phase, tt.err)
			if tt.isNil {
				assert.Nil(t, wrapped)
			} else {
				require.NotNil(t, wrapped)
				assert.Equal(t, tt.expected, wrapped.Error())
				// Verify error unwrapping works for sentinel errors
				if errors.Is(tt.err, ErrGameOver) {
					assert.True(t, errors.Is(wrapped, ErrGameOver))
				}
			}
		})
	}
}

func TestWrapPlayerError(t *testing.T) {
	tests := []struct {
		name      string
		playerID  int
		operation string
		err       error
		expected  string
		isNil     bool
	}{
		{
			name:      "nil error returns nil",
			playerID:  1,
			operation: "move validation",
			err:       nil,
			expected:  "",
			isNil:     true,
		},
		{
			name:      "player move validation error",
			playerID:  3,
			operation: "move validation",
			err:       ErrInsufficientArmy,
			expected:  "player 3 move validation: insufficient army to move",
		},
		{
			name:      "player action processing error",
			playerID:  0,
			operation: "action processing",
			err:       ErrInvalidPlayer,
			expected:  "player 0 action processing: invalid player ID",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wrapped := WrapPlayerError(tt.playerID, tt.operation, tt.err)
			if tt.isNil {
				assert.Nil(t, wrapped)
			} else {
				require.NotNil(t, wrapped)
				assert.Equal(t, tt.expected, wrapped.Error())
				// Verify error unwrapping works
				assert.True(t, errors.Is(wrapped, tt.err))
			}
		})
	}
}

func TestGameError(t *testing.T) {
	t.Run("with player ID", func(t *testing.T) {
		err := NewGameError(150, 2, "capture general", ErrNotOwned)
		assert.Equal(t, "turn 150: player 2 capture general: tile not owned by player", err.Error())
		// Verify unwrapping works
		assert.True(t, errors.Is(err, ErrNotOwned))
	})

	t.Run("without player ID", func(t *testing.T) {
		err := NewGameError(200, 0, "win condition check", ErrGameOver)
		assert.Equal(t, "turn 200: win condition check: game is over", err.Error())
		// Verify unwrapping works
		assert.True(t, errors.Is(err, ErrGameOver))
	})

	t.Run("errors.As functionality", func(t *testing.T) {
		originalErr := fmt.Errorf("network timeout")
		gameErr := NewGameError(50, 1, "network sync", originalErr)

		var extracted *GameError
		assert.True(t, errors.As(gameErr, &extracted))
		assert.Equal(t, 50, extracted.Turn)
		assert.Equal(t, 1, extracted.PlayerID)
		assert.Equal(t, "network sync", extracted.Operation)
	})
}

// TestErrorUtilitiesUsageExample demonstrates how to use these utilities in practice
func TestErrorUtilitiesUsageExample(t *testing.T) {
	// Example 1: Wrapping a move validation error
	board := NewBoard(10, 10)
	move := &MoveAction{PlayerID: 1, FromX: 5, FromY: 3, ToX: 5, ToY: 4}
	if err := move.Validate(board, 1); err != nil {
		wrappedErr := WrapActionError(move, err)
		fmt.Println(wrappedErr)
		// Output would be: player 1: move from (5,3) to (5,4): [error message]
	}

	// Example 2: Wrapping a game state error
	turn := 100
	if err := processGameTurn(); err != nil {
		wrappedErr := WrapGameStateError(turn, "action processing", err)
		fmt.Println(wrappedErr)
		// Output would be: game turn 100 [action processing]: [error message]
	}

	// Example 3: Using structured GameError
	gameErr := NewGameError(50, 2, "general capture", ErrNotOwned)
	fmt.Println(gameErr)
	// Output would be: turn 50: player 2 general capture: tile not owned by player

	// Example 4: Error checking with wrapped errors
	err := WrapPlayerError(1, "move", ErrInsufficientArmy)
	if errors.Is(err, ErrInsufficientArmy) {
		// This will be true - wrapped errors preserve the original error chain
		fmt.Println("Player doesn't have enough army")
	}
}

// Mock function for example
func processGameTurn() error {
	return fmt.Errorf("simulated error")
}
