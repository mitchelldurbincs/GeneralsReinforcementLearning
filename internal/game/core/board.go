package core 

// Tile represents a single cell on the map.
// Owner: -1 means neutral; 0..N-1 are player IDs.
// Army: number of units on that tile.
// Type - 0 = normal, 1 = general, 2 = city.
type Tile struct {
    Owner   int
    Army    int
    Type    int
    VisibleBitfield uint32 // Bit i = 1 if player i can see this tile (supports up to 32 players)
}

type Board struct {
    W, H int
    T    []Tile // length = W*H (row‑major)
}

const (
    TileNormal   = 0
    TileGeneral  = 1
    TileCity     = 2
	TileMountain = 3
    NeutralID    = -1
)

func (t *Tile) IsNeutral() bool { return t.Owner == NeutralID }
func (t *Tile) IsCity() bool    { return t.Type == TileCity }
func (t *Tile) IsGeneral() bool { return t.Type == TileGeneral }
func (t *Tile) IsMountain() bool { return t.Type == TileMountain }
func (t *Tile) IsEmpty() bool    { return t.IsNeutral() && t.Type == TileNormal && t.Army == 0 }

// New bitfield visibility methods
func (t *Tile) IsVisibleTo(playerID int) bool {
    if playerID < 0 || playerID >= 32 {
        return false
    }
    return t.VisibleBitfield & (1 << uint(playerID)) != 0
}

func (t *Tile) SetVisible(playerID int, visible bool) {
    if playerID < 0 || playerID >= 32 {
        return
    }
    if visible {
        t.VisibleBitfield |= (1 << uint(playerID))
    } else {
        t.VisibleBitfield &^= (1 << uint(playerID))
    }
}


func NewBoard(w, h int) *Board {
	b := &Board{W: w, H: h, T: make([]Tile, w*h)}
	for i := range b.T {
		// All tiles start neural and normal
		b.T[i].Owner = -1
		b.T[i].Type = TileNormal
		// Visibility is handled by VisibleBitfield, initialized to 0
	}
	return b
}

func (b *Board) Idx(x, y int) int       { return y*b.W + x }
func (b *Board) XY(idx int) (int, int)  { return idx % b.W, idx / b.W }

// InBounds checks if coordinates are within board boundaries
func (b *Board) InBounds(x, y int) bool {
	return x >= 0 && x < b.W && y >= 0 && y < b.H
}

// GetTile safely returns a tile pointer if coordinates are valid, nil otherwise
func (b *Board) GetTile(x, y int) *Tile {
	if !b.InBounds(x, y) {
		return nil
	}
	return &b.T[b.Idx(x, y)]
}

func (b *Board) Distance(x1, y1, x2, y2 int) int {
	dx := x1 - x2
	if dx < 0 {
		dx = -dx
	}
	dy := y1 - y2
	if dy < 0 {
		dy = -dy
	}
	return dx + dy
}