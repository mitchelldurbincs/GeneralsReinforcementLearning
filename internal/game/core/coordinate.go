package core

import "fmt"

// Coordinate represents a position on the game board
type Coordinate struct {
	X, Y int
}

// NewCoordinate creates a new coordinate with the given x and y values
func NewCoordinate(x, y int) Coordinate {
	return Coordinate{X: x, Y: y}
}

// FromIndex creates a coordinate from a board array index using row-major ordering
func FromIndex(idx, width int) Coordinate {
	return Coordinate{
		X: idx % width,
		Y: idx / width,
	}
}

// IsValid checks if the coordinate is within the given bounds
func (c Coordinate) IsValid(width, height int) bool {
	return c.X >= 0 && c.X < width && c.Y >= 0 && c.Y < height
}

// ToIndex converts the coordinate to a board array index using row-major ordering
func (c Coordinate) ToIndex(width int) int {
	return c.Y*width + c.X
}

// DistanceTo calculates the Manhattan distance to another coordinate
func (c Coordinate) DistanceTo(other Coordinate) int {
	dx := c.X - other.X
	dy := c.Y - other.Y
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}
	return dx + dy
}

// IsAdjacentTo checks if this coordinate is orthogonally adjacent to another
func (c Coordinate) IsAdjacentTo(other Coordinate) bool {
	dx := c.X - other.X
	dy := c.Y - other.Y
	
	// Must be exactly one step away in either X or Y direction, but not both
	return (dx == 0 && (dy == 1 || dy == -1)) || (dy == 0 && (dx == 1 || dx == -1))
}

// Neighbors returns the four orthogonal neighbors of this coordinate
func (c Coordinate) Neighbors() []Coordinate {
	return []Coordinate{
		{X: c.X, Y: c.Y - 1}, // North
		{X: c.X + 1, Y: c.Y}, // East  
		{X: c.X, Y: c.Y + 1}, // South
		{X: c.X - 1, Y: c.Y}, // West
	}
}

// ValidNeighbors returns only the neighbors that are within the given bounds
func (c Coordinate) ValidNeighbors(width, height int) []Coordinate {
	neighbors := c.Neighbors()
	valid := make([]Coordinate, 0, 4)
	
	for _, n := range neighbors {
		if n.IsValid(width, height) {
			valid = append(valid, n)
		}
	}
	
	return valid
}

// Add returns a new coordinate that is the sum of this coordinate and another
func (c Coordinate) Add(other Coordinate) Coordinate {
	return Coordinate{
		X: c.X + other.X,
		Y: c.Y + other.Y,
	}
}

// Sub returns a new coordinate that is the difference between this coordinate and another
func (c Coordinate) Sub(other Coordinate) Coordinate {
	return Coordinate{
		X: c.X - other.X,
		Y: c.Y - other.Y,
	}
}

// Equal checks if two coordinates are equal
func (c Coordinate) Equal(other Coordinate) bool {
	return c.X == other.X && c.Y == other.Y
}

// String returns a string representation of the coordinate
func (c Coordinate) String() string {
	return fmt.Sprintf("(%d,%d)", c.X, c.Y)
}

// Direction represents a cardinal direction
type Direction int

const (
	North Direction = iota
	East
	South
	West
)

// DirectionVectors provides coordinate offsets for each direction
var DirectionVectors = map[Direction]Coordinate{
	North: {X: 0, Y: -1},
	East:  {X: 1, Y: 0},
	South: {X: 0, Y: 1},
	West:  {X: -1, Y: 0},
}

// Move returns a new coordinate moved one step in the given direction
func (c Coordinate) Move(direction Direction) Coordinate {
	if offset, ok := DirectionVectors[direction]; ok {
		return c.Add(offset)
	}
	return c
}

// DirectionTo returns the direction from this coordinate to an adjacent coordinate
// Returns -1 if the coordinates are not adjacent
func (c Coordinate) DirectionTo(other Coordinate) Direction {
	if !c.IsAdjacentTo(other) {
		return -1
	}
	
	dx := other.X - c.X
	dy := other.Y - c.Y
	
	switch {
	case dy == -1:
		return North
	case dx == 1:
		return East
	case dy == 1:
		return South
	case dx == -1:
		return West
	default:
		return -1
	}
}