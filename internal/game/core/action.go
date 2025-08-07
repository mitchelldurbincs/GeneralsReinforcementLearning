package core

import (
	"fmt"
)

// ActionType represents the type of action
type ActionType int

const (
	ActionMove ActionType = iota
	// Future: ActionSplit, ActionDefend?, etc.
)

// Action represents a player action
type Action interface {
	GetPlayerID() int
	GetType() ActionType
	Validate(b *Board, playerID int) error
}

// MoveAction represents moving armies from one tile to another
type MoveAction struct {
	PlayerID int
	// Keep existing fields for backward compatibility
	FromX int
	FromY int
	ToX   int
	ToY   int
	// New coordinate fields
	From Coordinate
	To   Coordinate
	// If true, move all but 1 army. If false, move half (integer division, minimum 1 if source army > 1)
	MoveAll bool
}

func (m *MoveAction) GetPlayerID() int    { return m.PlayerID }
func (m *MoveAction) GetType() ActionType { return ActionMove }

// GetFrom returns the From coordinate, using the new field if set, otherwise converting from X,Y
func (m *MoveAction) GetFrom() Coordinate {
	if m.From != (Coordinate{}) {
		return m.From
	}
	return Coordinate{X: m.FromX, Y: m.FromY}
}

// GetTo returns the To coordinate, using the new field if set, otherwise converting from X,Y
func (m *MoveAction) GetTo() Coordinate {
	if m.To != (Coordinate{}) {
		return m.To
	}
	return Coordinate{X: m.ToX, Y: m.ToY}
}

func (m *MoveAction) Validate(b *Board, playerID int) error {
	// Check bounds for From coordinates
	if !b.InBounds(m.FromX, m.FromY) {
		return fmt.Errorf("player %d: move from (%d,%d) out of bounds: %w", m.PlayerID, m.FromX, m.FromY, ErrInvalidCoordinates)
	}
	// Check bounds for To coordinates
	if !b.InBounds(m.ToX, m.ToY) {
		return fmt.Errorf("player %d: move to (%d,%d) out of bounds: %w", m.PlayerID, m.ToX, m.ToY, ErrInvalidCoordinates)
	}

	// Check if From and To are the same tile
	if m.FromX == m.ToX && m.FromY == m.ToY {
		return fmt.Errorf("player %d: move from/to same tile (%d,%d): %w", m.PlayerID, m.FromX, m.FromY, ErrMoveToSelf)
	}

	// Check adjacency (only orthogonal moves allowed)
	from := m.GetFrom()
	to := m.GetTo()
	if !from.IsAdjacentTo(to) {
		return fmt.Errorf("player %d: move from (%d,%d) to (%d,%d) not adjacent: %w", m.PlayerID, m.FromX, m.FromY, m.ToX, m.ToY, ErrNotAdjacent)
	}

	fromIdx := b.Idx(m.FromX, m.FromY)
	fromTile := &b.T[fromIdx]

	// Check ownership of the FromTile
	if fromTile.Owner != playerID {
		return fmt.Errorf("player %d: tile at (%d,%d) owned by player %d: %w", m.PlayerID, m.FromX, m.FromY, fromTile.Owner, ErrNotOwned)
	}

	// Check if FromTile has armies to move
	if fromTile.Army <= 1 {
		return fmt.Errorf("player %d: tile at (%d,%d) has only %d army: %w", m.PlayerID, m.FromX, m.FromY, fromTile.Army, ErrInsufficientArmy)
	}

	// Check properties of the ToTile
	toIdx := b.Idx(m.ToX, m.ToY)
	toTile := &b.T[toIdx]

	// Check if ToTile is a mountain
	if toTile.IsMountain() { // Assumes Tile struct has IsMountain() method
		return fmt.Errorf("player %d: cannot move to mountain at (%d,%d): %w", m.PlayerID, m.ToX, m.ToY, ErrTargetIsMountain)
	}

	// Add any other ToTile validations here if needed in the future
	// For example, if certain tile types cannot be attacked/entered.
	// For now, only mountains are explicitly restricted as per your request.

	return nil
}
