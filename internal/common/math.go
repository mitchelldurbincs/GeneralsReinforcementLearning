package common

import "github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"

// Abs returns the absolute value of an integer
func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Min returns the minimum of two integers
func Min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Max returns the maximum of two integers
func Max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// DistanceCoord calculates the Manhattan distance between two coordinates
func DistanceCoord(from, to core.Coordinate) int {
	return from.DistanceTo(to)
}

// IsAdjacentCoord checks if two coordinates are orthogonally adjacent (not diagonally)
func IsAdjacentCoord(from, to core.Coordinate) bool {
	return from.IsAdjacentTo(to)
}
