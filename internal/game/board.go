package game

// Tile represents a single cell on the map.
// Owner: -1 means neutral; 0..N-1 are player IDs.
// Army: number of units on that tile.
// Type - 0 = normal, 1 = general, 2 = city.
type Tile struct {
    Owner int
    Army  int
    Type  int 
}

type Board struct {
    W, H int
    T    []Tile // length = W*H (row‑major)
}

const (
    TileNormal  = 0
    TileGeneral = 1
    TileCity    = 2
    NeutralID   = -1
)

func (t *Tile) IsNeutral() bool { return t.Owner == NeutralID }
func (t *Tile) IsCity() bool    { return t.Type == TileCity }
func (t *Tile) IsGeneral() bool { return t.Type == TileGeneral }
func (t Tile) IsEmpty() bool    { return t.IsNeutral() && t.Type == TileNormal && t.Army == 0 }

func NewBoard(w, h int) *Board {
	b := &Board{W: w, H: h, T: make([]Tile, w*h)}
	for i := range b.T {
		b.T[i].Owner = -1
	}
	return b
}

func (b *Board) Idx(x, y int) int       { return y*b.W + x }
func (b *Board) XY(idx int) (int, int)  { return idx % b.W, idx / b.W }

// Manhattan distance between two board coordinates
func (b *Board) Distance(x1, y1, x2, y2 int) int {
	if x1 < x2 {
		x2, x1 = x1, x2
	}
	if y1 < y2 {
		y2, y1 = y1, y2
	}
	return (x1 - x2) + (y1 - y2)
}