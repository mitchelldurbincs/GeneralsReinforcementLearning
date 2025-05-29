package core

// ActionType represents the type of action
type ActionType int

const (
	ActionMove ActionType = iota
	// Future: ActionSplit, ActionDefend, etc.
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
	FromX    int
	FromY    int
	ToX      int
	ToY      int
	// If true, move all but 1 army. If false, move half (rounded up)
	MoveAll  bool
}

func (m *MoveAction) GetPlayerID() int    { return m.PlayerID }
func (m *MoveAction) GetType() ActionType { return ActionMove }

func (m *MoveAction) Validate(b *Board, playerID int) error {
	// Check bounds
	if m.FromX < 0 || m.FromX >= b.W || m.FromY < 0 || m.FromY >= b.H {
		return ErrInvalidCoordinates
	}
	if m.ToX < 0 || m.ToX >= b.W || m.ToY < 0 || m.ToY >= b.H {
		return ErrInvalidCoordinates
	}
	
	// Check adjacency (only orthogonal moves allowed)
	if !IsAdjacent(m.FromX, m.FromY, m.ToX, m.ToY) {
		return ErrNotAdjacent
	}
	
	fromIdx := b.Idx(m.FromX, m.FromY)
	fromTile := &b.T[fromIdx]
	
	// Check ownership
	if fromTile.Owner != playerID {
		return ErrNotOwned
	}
	
	// Check has armies to move
	if fromTile.Army <= 1 {
		return ErrInsufficientArmy
	}
	
	return nil
}

// IsAdjacent checks if two coordinates are orthogonally adjacent
func IsAdjacent(x1, y1, x2, y2 int) bool {
	dx := abs(x1 - x2)
	dy := abs(y1 - y2)
	return (dx == 1 && dy == 0) || (dx == 0 && dy == 1)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}