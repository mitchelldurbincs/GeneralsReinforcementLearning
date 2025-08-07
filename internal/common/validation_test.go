package common

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsValidCoordinate(t *testing.T) {
	tests := []struct {
		name     string
		x, y     int
		width    int
		height   int
		expected bool
	}{
		// Valid coordinates
		{"top-left corner", 0, 0, 10, 10, true},
		{"top-right corner", 9, 0, 10, 10, true},
		{"bottom-left corner", 0, 9, 10, 10, true},
		{"bottom-right corner", 9, 9, 10, 10, true},
		{"center", 5, 5, 10, 10, true},
		{"edge cases", 0, 0, 1, 1, true},

		// Invalid coordinates
		{"negative x", -1, 5, 10, 10, false},
		{"negative y", 5, -1, 10, 10, false},
		{"x equals width", 10, 5, 10, 10, false},
		{"y equals height", 5, 10, 10, 10, false},
		{"x greater than width", 15, 5, 10, 10, false},
		{"y greater than height", 5, 15, 10, 10, false},
		{"both negative", -1, -1, 10, 10, false},
		{"both out of bounds", 20, 20, 10, 10, false},

		// Zero dimensions
		{"zero width", 0, 0, 0, 10, false},
		{"zero height", 0, 0, 10, 0, false},
		{"zero both", 0, 0, 0, 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsValidCoordinate(tt.x, tt.y, tt.width, tt.height)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsAdjacent(t *testing.T) {
	tests := []struct {
		name     string
		x1, y1   int
		x2, y2   int
		expected bool
	}{
		// Adjacent cases
		{"right adjacent", 5, 5, 6, 5, true},
		{"left adjacent", 5, 5, 4, 5, true},
		{"down adjacent", 5, 5, 5, 6, true},
		{"up adjacent", 5, 5, 5, 4, true},
		{"edge case adjacent", 0, 0, 1, 0, true},
		{"negative coords adjacent", -1, 0, 0, 0, true},

		// Non-adjacent cases
		{"same position", 5, 5, 5, 5, false},
		{"diagonal", 5, 5, 6, 6, false},
		{"two steps horizontal", 5, 5, 7, 5, false},
		{"two steps vertical", 5, 5, 5, 7, false},
		{"far away", 1, 1, 10, 10, false},
		{"diagonal close", 5, 5, 4, 4, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsAdjacent(tt.x1, tt.y1, tt.x2, tt.y2)
			assert.Equal(t, tt.expected, result)

			// Test symmetry
			resultReverse := IsAdjacent(tt.x2, tt.y2, tt.x1, tt.y1)
			assert.Equal(t, result, resultReverse, "adjacency should be symmetric")
		})
	}
}

func TestManhattanDistance(t *testing.T) {
	tests := []struct {
		name     string
		x1, y1   int
		x2, y2   int
		expected int
	}{
		{"same position", 5, 5, 5, 5, 0},
		{"horizontal distance", 0, 0, 5, 0, 5},
		{"vertical distance", 0, 0, 0, 5, 5},
		{"diagonal", 0, 0, 3, 4, 7},
		{"negative coords", -2, -3, 1, 2, 8},
		{"mixed positive negative", -5, 5, 5, -5, 20},
		{"adjacent", 5, 5, 6, 5, 1},
		{"large distance", 0, 0, 100, 100, 200},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ManhattanDistance(tt.x1, tt.y1, tt.x2, tt.y2)
			assert.Equal(t, tt.expected, result)

			// Test symmetry
			resultReverse := ManhattanDistance(tt.x2, tt.y2, tt.x1, tt.y1)
			assert.Equal(t, result, resultReverse, "distance should be symmetric")
		})
	}
}

func TestValidationIntegration(t *testing.T) {
	// Test that IsAdjacent implies ManhattanDistance of 1
	t.Run("adjacent implies distance 1", func(t *testing.T) {
		testCases := []struct {
			x1, y1, x2, y2 int
		}{
			{5, 5, 6, 5},
			{0, 0, 0, 1},
			{10, 10, 9, 10},
		}

		for _, tc := range testCases {
			if IsAdjacent(tc.x1, tc.y1, tc.x2, tc.y2) {
				dist := ManhattanDistance(tc.x1, tc.y1, tc.x2, tc.y2)
				assert.Equal(t, 1, dist)
			}
		}
	})

	// Test that non-adjacent with distance 1 is impossible
	t.Run("distance 1 implies adjacent", func(t *testing.T) {
		for x1 := -2; x1 <= 2; x1++ {
			for y1 := -2; y1 <= 2; y1++ {
				for x2 := -2; x2 <= 2; x2++ {
					for y2 := -2; y2 <= 2; y2++ {
						dist := ManhattanDistance(x1, y1, x2, y2)
						isAdj := IsAdjacent(x1, y1, x2, y2)
						if dist == 1 {
							assert.True(t, isAdj, "distance 1 should imply adjacent")
						}
						if isAdj {
							assert.Equal(t, 1, dist, "adjacent should imply distance 1")
						}
					}
				}
			}
		}
	})
}

// Benchmarks
func BenchmarkIsValidCoordinate(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = IsValidCoordinate(5, 5, 10, 10)
	}
}

func BenchmarkIsAdjacent(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = IsAdjacent(5, 5, 6, 5)
	}
}

func BenchmarkManhattanDistance(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = ManhattanDistance(0, 0, 10, 10)
	}
}
