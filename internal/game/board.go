package game

// Tile represents a single cell on the map.
// TODO: add owner, army size, city/fort flags, etc.
type Tile struct {
    Owner int  // -1 means neutral
    Army  int
}

type Board struct {
    W, H int
    T    []Tile // length = W*H (row‑major)
}

func NewBoard(w, h int) *Board {
    b := &Board{W: w, H: h, T: make([]Tile, w*h)}
    return b
}

func (b *Board) Idx(x, y int) int { return y*b.W + x }