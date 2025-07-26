package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewCoordinate(t *testing.T) {
	c := NewCoordinate(3, 5)
	assert.Equal(t, 3, c.X)
	assert.Equal(t, 5, c.Y)
}

func TestCoordinate_FromIndex(t *testing.T) {
	tests := []struct {
		name     string
		index    int
		width    int
		expected Coordinate
	}{
		{"TopLeft", 0, 10, Coordinate{0, 0}},
		{"TopRight", 9, 10, Coordinate{9, 0}},
		{"SecondRow", 10, 10, Coordinate{0, 1}},
		{"Middle", 55, 10, Coordinate{5, 5}},
		{"BottomRight", 99, 10, Coordinate{9, 9}},
		{"SmallBoard", 7, 4, Coordinate{3, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FromIndex(tt.index, tt.width)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCoordinate_ToIndex(t *testing.T) {
	tests := []struct {
		name     string
		coord    Coordinate
		width    int
		expected int
	}{
		{"TopLeft", Coordinate{0, 0}, 10, 0},
		{"TopRight", Coordinate{9, 0}, 10, 9},
		{"SecondRow", Coordinate{0, 1}, 10, 10},
		{"Middle", Coordinate{5, 5}, 10, 55},
		{"BottomRight", Coordinate{9, 9}, 10, 99},
		{"SmallBoard", Coordinate{3, 1}, 4, 7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.coord.ToIndex(tt.width)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCoordinate_RoundTrip(t *testing.T) {
	// Test that FromIndex and ToIndex are inverses
	width := 10
	for i := 0; i < 100; i++ {
		coord := FromIndex(i, width)
		index := coord.ToIndex(width)
		assert.Equal(t, i, index, "Round trip failed for index %d", i)
	}
}

func TestCoordinate_IsValid(t *testing.T) {
	tests := []struct {
		name   string
		coord  Coordinate
		width  int
		height int
		valid  bool
	}{
		{"Valid_Origin", Coordinate{0, 0}, 10, 10, true},
		{"Valid_Middle", Coordinate{5, 5}, 10, 10, true},
		{"Valid_Edge", Coordinate{9, 9}, 10, 10, true},
		{"Invalid_NegativeX", Coordinate{-1, 5}, 10, 10, false},
		{"Invalid_NegativeY", Coordinate{5, -1}, 10, 10, false},
		{"Invalid_TooLargeX", Coordinate{10, 5}, 10, 10, false},
		{"Invalid_TooLargeY", Coordinate{5, 10}, 10, 10, false},
		{"Invalid_BothNegative", Coordinate{-1, -1}, 10, 10, false},
		{"Invalid_BothTooLarge", Coordinate{10, 10}, 10, 10, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.coord.IsValid(tt.width, tt.height)
			assert.Equal(t, tt.valid, result)
		})
	}
}

func TestCoordinate_DistanceTo(t *testing.T) {
	tests := []struct {
		name     string
		from     Coordinate
		to       Coordinate
		expected int
	}{
		{"Same", Coordinate{5, 5}, Coordinate{5, 5}, 0},
		{"Adjacent_Horizontal", Coordinate{5, 5}, Coordinate{6, 5}, 1},
		{"Adjacent_Vertical", Coordinate{5, 5}, Coordinate{5, 6}, 1},
		{"Diagonal", Coordinate{0, 0}, Coordinate{1, 1}, 2},
		{"Far", Coordinate{0, 0}, Coordinate{5, 7}, 12},
		{"Negative", Coordinate{-2, -3}, Coordinate{2, 3}, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.from.DistanceTo(tt.to)
			assert.Equal(t, tt.expected, result)
			// Distance should be symmetric
			reverse := tt.to.DistanceTo(tt.from)
			assert.Equal(t, tt.expected, reverse, "Distance not symmetric")
		})
	}
}

func TestCoordinate_IsAdjacentTo(t *testing.T) {
	center := Coordinate{5, 5}
	tests := []struct {
		name     string
		other    Coordinate
		adjacent bool
	}{
		{"North", Coordinate{5, 4}, true},
		{"East", Coordinate{6, 5}, true},
		{"South", Coordinate{5, 6}, true},
		{"West", Coordinate{4, 5}, true},
		{"NorthEast", Coordinate{6, 4}, false},
		{"SouthEast", Coordinate{6, 6}, false},
		{"SouthWest", Coordinate{4, 6}, false},
		{"NorthWest", Coordinate{4, 4}, false},
		{"Same", Coordinate{5, 5}, false},
		{"TwoAway", Coordinate{7, 5}, false},
		{"FarAway", Coordinate{0, 0}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := center.IsAdjacentTo(tt.other)
			assert.Equal(t, tt.adjacent, result)
			// Adjacency should be symmetric
			reverse := tt.other.IsAdjacentTo(center)
			assert.Equal(t, tt.adjacent, reverse, "Adjacency not symmetric")
		})
	}
}

func TestCoordinate_Neighbors(t *testing.T) {
	c := Coordinate{5, 5}
	neighbors := c.Neighbors()
	
	assert.Len(t, neighbors, 4)
	assert.Contains(t, neighbors, Coordinate{5, 4}) // North
	assert.Contains(t, neighbors, Coordinate{6, 5}) // East
	assert.Contains(t, neighbors, Coordinate{5, 6}) // South
	assert.Contains(t, neighbors, Coordinate{4, 5}) // West
}

func TestCoordinate_ValidNeighbors(t *testing.T) {
	tests := []struct {
		name          string
		coord         Coordinate
		width, height int
		expectedCount int
	}{
		{"Center", Coordinate{5, 5}, 10, 10, 4},
		{"TopLeft", Coordinate{0, 0}, 10, 10, 2},
		{"TopRight", Coordinate{9, 0}, 10, 10, 2},
		{"BottomLeft", Coordinate{0, 9}, 10, 10, 2},
		{"BottomRight", Coordinate{9, 9}, 10, 10, 2},
		{"TopEdge", Coordinate{5, 0}, 10, 10, 3},
		{"BottomEdge", Coordinate{5, 9}, 10, 10, 3},
		{"LeftEdge", Coordinate{0, 5}, 10, 10, 3},
		{"RightEdge", Coordinate{9, 5}, 10, 10, 3},
		{"SingleCell", Coordinate{0, 0}, 1, 1, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := tt.coord.ValidNeighbors(tt.width, tt.height)
			assert.Len(t, valid, tt.expectedCount)
			
			// All returned neighbors should be valid
			for _, n := range valid {
				assert.True(t, n.IsValid(tt.width, tt.height))
				assert.True(t, n.IsAdjacentTo(tt.coord))
			}
		})
	}
}

func TestCoordinate_Add(t *testing.T) {
	c1 := Coordinate{3, 4}
	c2 := Coordinate{2, -1}
	result := c1.Add(c2)
	assert.Equal(t, Coordinate{5, 3}, result)
	
	// Original should be unchanged
	assert.Equal(t, Coordinate{3, 4}, c1)
	assert.Equal(t, Coordinate{2, -1}, c2)
}

func TestCoordinate_Sub(t *testing.T) {
	c1 := Coordinate{5, 3}
	c2 := Coordinate{2, -1}
	result := c1.Sub(c2)
	assert.Equal(t, Coordinate{3, 4}, result)
	
	// Original should be unchanged
	assert.Equal(t, Coordinate{5, 3}, c1)
	assert.Equal(t, Coordinate{2, -1}, c2)
}

func TestCoordinate_Equal(t *testing.T) {
	tests := []struct {
		c1, c2 Coordinate
		equal  bool
	}{
		{Coordinate{0, 0}, Coordinate{0, 0}, true},
		{Coordinate{5, 7}, Coordinate{5, 7}, true},
		{Coordinate{-1, -1}, Coordinate{-1, -1}, true},
		{Coordinate{5, 7}, Coordinate{7, 5}, false},
		{Coordinate{0, 0}, Coordinate{0, 1}, false},
		{Coordinate{0, 0}, Coordinate{1, 0}, false},
	}

	for _, tt := range tests {
		result := tt.c1.Equal(tt.c2)
		assert.Equal(t, tt.equal, result)
		// Equality should be symmetric
		reverse := tt.c2.Equal(tt.c1)
		assert.Equal(t, tt.equal, reverse)
	}
}

func TestCoordinate_String(t *testing.T) {
	tests := []struct {
		coord    Coordinate
		expected string
	}{
		{Coordinate{0, 0}, "(0,0)"},
		{Coordinate{5, 7}, "(5,7)"},
		{Coordinate{-1, -2}, "(-1,-2)"},
		{Coordinate{100, 200}, "(100,200)"},
	}

	for _, tt := range tests {
		result := tt.coord.String()
		assert.Equal(t, tt.expected, result)
	}
}

func TestCoordinate_Move(t *testing.T) {
	start := Coordinate{5, 5}
	tests := []struct {
		name      string
		direction Direction
		expected  Coordinate
	}{
		{"North", North, Coordinate{5, 4}},
		{"East", East, Coordinate{6, 5}},
		{"South", South, Coordinate{5, 6}},
		{"West", West, Coordinate{4, 5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := start.Move(tt.direction)
			assert.Equal(t, tt.expected, result)
			// Original should be unchanged
			assert.Equal(t, Coordinate{5, 5}, start)
		})
	}
}

func TestCoordinate_DirectionTo(t *testing.T) {
	center := Coordinate{5, 5}
	tests := []struct {
		name      string
		other     Coordinate
		expected  Direction
	}{
		{"North", Coordinate{5, 4}, North},
		{"East", Coordinate{6, 5}, East},
		{"South", Coordinate{5, 6}, South},
		{"West", Coordinate{4, 5}, West},
		{"NotAdjacent_Diagonal", Coordinate{6, 6}, -1},
		{"NotAdjacent_Same", Coordinate{5, 5}, -1},
		{"NotAdjacent_Far", Coordinate{10, 10}, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := center.DirectionTo(tt.other)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDirectionVectors(t *testing.T) {
	// Verify all directions are defined
	assert.Len(t, DirectionVectors, 4)
	
	// Verify vectors are correct
	assert.Equal(t, Coordinate{0, -1}, DirectionVectors[North])
	assert.Equal(t, Coordinate{1, 0}, DirectionVectors[East])
	assert.Equal(t, Coordinate{0, 1}, DirectionVectors[South])
	assert.Equal(t, Coordinate{-1, 0}, DirectionVectors[West])
	
	// Verify all vectors have length 1
	for dir, vec := range DirectionVectors {
		length := vec.DistanceTo(Coordinate{0, 0})
		assert.Equal(t, 1, length, "Direction %v vector has wrong length", dir)
	}
}

func TestCoordinate_ComparableAsMapKey(t *testing.T) {
	// Test that Coordinate can be used as a map key
	m := make(map[Coordinate]string)
	
	c1 := Coordinate{5, 5}
	c2 := Coordinate{5, 5}
	c3 := Coordinate{6, 5}
	
	m[c1] = "first"
	m[c3] = "third"
	
	// Same coordinates should map to same key
	assert.Equal(t, "first", m[c2])
	assert.Equal(t, "third", m[c3])
	assert.Len(t, m, 2)
}

func BenchmarkCoordinate_ToIndex(b *testing.B) {
	c := Coordinate{50, 50}
	width := 100
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c.ToIndex(width)
	}
}

func BenchmarkCoordinate_DistanceTo(b *testing.B) {
	c1 := Coordinate{0, 0}
	c2 := Coordinate{50, 50}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c1.DistanceTo(c2)
	}
}

func BenchmarkCoordinate_ValidNeighbors(b *testing.B) {
	c := Coordinate{50, 50}
	width, height := 100, 100
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = c.ValidNeighbors(width, height)
	}
}