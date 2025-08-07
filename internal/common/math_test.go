package common

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAbs(t *testing.T) {
	tests := []struct {
		name     string
		input    int
		expected int
	}{
		{"positive number", 5, 5},
		{"negative number", -5, 5},
		{"zero", 0, 0},
		{"large positive", 1000000, 1000000},
		{"large negative", -1000000, 1000000},
		{"min int special case", math.MinInt32 + 1, math.MaxInt32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Abs(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestMin(t *testing.T) {
	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"a smaller", 3, 5, 3},
		{"b smaller", 7, 2, 2},
		{"equal", 4, 4, 4},
		{"negative numbers", -5, -3, -5},
		{"positive and negative", 5, -3, -3},
		{"zero and positive", 0, 10, 0},
		{"zero and negative", 0, -10, -10},
		{"large numbers", 1000000, 999999, 999999},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Min(tt.a, tt.b)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestMax(t *testing.T) {
	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"a larger", 5, 3, 5},
		{"b larger", 2, 7, 7},
		{"equal", 4, 4, 4},
		{"negative numbers", -5, -3, -3},
		{"positive and negative", 5, -3, 5},
		{"zero and positive", 0, 10, 10},
		{"zero and negative", 0, -10, 0},
		{"large numbers", 1000000, 999999, 1000000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Max(tt.a, tt.b)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestMathFunctionsConsistency(t *testing.T) {
	// Test that Min and Max are consistent
	t.Run("min max consistency", func(t *testing.T) {
		testPairs := [][2]int{
			{5, 3},
			{-5, -3},
			{0, 10},
			{-10, 10},
			{7, 7},
		}

		for _, pair := range testPairs {
			a, b := pair[0], pair[1]
			minVal := Min(a, b)
			maxVal := Max(a, b)

			// Min should always be <= Max
			assert.LessOrEqual(t, minVal, maxVal)

			// One of them should be a, the other b (unless equal)
			if a != b {
				assert.True(t, (minVal == a && maxVal == b) || (minVal == b && maxVal == a))
			} else {
				assert.Equal(t, a, minVal)
				assert.Equal(t, a, maxVal)
			}
		}
	})

	// Test Abs properties
	t.Run("abs properties", func(t *testing.T) {
		values := []int{-10, -1, 0, 1, 10, 100}

		for _, v := range values {
			absV := Abs(v)
			// Abs value is always non-negative
			assert.GreaterOrEqual(t, absV, 0)

			// Abs(v) == Abs(-v)
			assert.Equal(t, Abs(v), Abs(-v))

			// If v >= 0, Abs(v) == v
			if v >= 0 {
				assert.Equal(t, v, absV)
			}
		}
	})
}

func TestMinMaxChain(t *testing.T) {
	// Test chaining Min/Max operations
	t.Run("find min of three", func(t *testing.T) {
		a, b, c := 5, 3, 7
		minOfThree := Min(Min(a, b), c)
		assert.Equal(t, 3, minOfThree)
	})

	t.Run("find max of three", func(t *testing.T) {
		a, b, c := 5, 3, 7
		maxOfThree := Max(Max(a, b), c)
		assert.Equal(t, 7, maxOfThree)
	})
}

// Edge cases
func TestMathEdgeCases(t *testing.T) {
	t.Run("min with max int", func(t *testing.T) {
		result := Min(math.MaxInt32, math.MaxInt32-1)
		assert.Equal(t, math.MaxInt32-1, result)
	})

	t.Run("max with min int", func(t *testing.T) {
		result := Max(math.MinInt32, math.MinInt32+1)
		assert.Equal(t, math.MinInt32+1, result)
	})
}

// Benchmarks
func BenchmarkAbs(b *testing.B) {
	values := []int{-5, 5, -100, 100, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Abs(values[i%len(values)])
	}
}

func BenchmarkMin(b *testing.B) {
	pairs := [][2]int{{5, 3}, {-5, -3}, {100, 200}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pair := pairs[i%len(pairs)]
		_ = Min(pair[0], pair[1])
	}
}

func BenchmarkMax(b *testing.B) {
	pairs := [][2]int{{5, 3}, {-5, -3}, {100, 200}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pair := pairs[i%len(pairs)]
		_ = Max(pair[0], pair[1])
	}
}
