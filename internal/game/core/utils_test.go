package core

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIntToStringFixedWidth(t *testing.T) {
	tests := []struct {
		name     string
		num      int
		width    int
		expected string
	}{
		{
			name:     "single digit with width 3",
			num:      5,
			width:    3,
			expected: "  5",
		},
		{
			name:     "two digits with width 3",
			num:      42,
			width:    3,
			expected: " 42",
		},
		{
			name:     "three digits with width 3",
			num:      123,
			width:    3,
			expected: "123",
		},
		{
			name:     "number exceeds width",
			num:      1234,
			width:    3,
			expected: "1234", // fmt.Sprintf doesn't truncate
		},
		{
			name:     "negative number",
			num:      -5,
			width:    3,
			expected: " -5",
		},
		{
			name:     "zero",
			num:      0,
			width:    3,
			expected: "  0",
		},
		{
			name:     "width of 1",
			num:      7,
			width:    1,
			expected: "7",
		},
		{
			name:     "large width",
			num:      99,
			width:    10,
			expected: "        99",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IntToStringFixedWidth(tt.num, tt.width)
			assert.Equal(t, tt.expected, result)
			assert.Equal(t, len(tt.expected), len(result))
		})
	}
}

func TestIntToStringFixedWidth_EdgeCases(t *testing.T) {
	t.Run("zero width", func(t *testing.T) {
		result := IntToStringFixedWidth(123, 0)
		assert.Equal(t, "123", result)
	})

	t.Run("negative width", func(t *testing.T) {
		result := IntToStringFixedWidth(123, -5)
		// fmt.Sprintf with negative width adds trailing spaces
		assert.Equal(t, "123  ", result)
	})

	t.Run("very large number", func(t *testing.T) {
		result := IntToStringFixedWidth(999999999, 5)
		assert.Equal(t, "999999999", result)
		assert.True(t, len(result) > 5)
	})
}

func TestGetActionType(t *testing.T) {
	tests := []struct {
		name     string
		action   Action
		expected string
	}{
		{
			name:     "nil action",
			action:   nil,
			expected: "nil",
		},
		{
			name: "MoveAction pointer",
			action: &MoveAction{
				PlayerID: 0,
				FromX:    0,
				FromY:    0,
				ToX:      1,
				ToY:      0,
				MoveAll:  true,
			},
			expected: "*core.MoveAction",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetActionType(tt.action)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetActionType_MultipleTypes(t *testing.T) {
	// Test that the function correctly identifies different action types
	moveAction := &MoveAction{}
	actionType := GetActionType(moveAction)
	assert.True(t, strings.Contains(actionType, "MoveAction"))
	assert.True(t, strings.Contains(actionType, "*"))

	// Test with interface holding a value
	var action Action = moveAction
	actionType = GetActionType(action)
	assert.True(t, strings.Contains(actionType, "MoveAction"))
}

// Benchmark tests to understand performance characteristics
func BenchmarkIntToStringFixedWidth(b *testing.B) {
	testCases := []struct {
		name  string
		num   int
		width int
	}{
		{"small number small width", 5, 3},
		{"large number small width", 12345, 3},
		{"small number large width", 5, 10},
		{"large number large width", 12345, 10},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = IntToStringFixedWidth(tc.num, tc.width)
			}
		})
	}
}

func BenchmarkGetActionType(b *testing.B) {
	action := &MoveAction{
		PlayerID: 0,
		FromX:    0,
		FromY:    0,
		ToX:      1,
		ToY:      0,
		MoveAll:  true,
	}

	b.Run("non-nil action", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = GetActionType(action)
		}
	})

	b.Run("nil action", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = GetActionType(nil)
		}
	})
}
