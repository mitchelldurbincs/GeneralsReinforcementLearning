package common

import "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"

// IsValidCoordinate checks if the given coordinates are within the bounds of the board
func IsValidCoordinate(x, y, width, height int) bool {
	return x >= 0 && x < width && y >= 0 && y < height
}

// IsValidCoordinateStruct checks if the given coordinate struct is within the bounds of the board
func IsValidCoordinateStruct(c core.Coordinate, width, height int) bool {
	return c.IsValid(width, height)
}

// IsAdjacent checks if two positions are orthogonally adjacent (not diagonally)
func IsAdjacent(x1, y1, x2, y2 int) bool {
	dx := Abs(x1 - x2)
	dy := Abs(y1 - y2)
	return (dx == 1 && dy == 0) || (dx == 0 && dy == 1)
}

// ManhattanDistance calculates the Manhattan distance between two points
func ManhattanDistance(x1, y1, x2, y2 int) int {
	return Abs(x1-x2) + Abs(y1-y2)
}
