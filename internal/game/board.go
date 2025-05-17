package game

// Tile represents a single cell on the map.
// Owner: -1 means neutral; 0..N-1 are player IDs.
// Army: number of units on that tile.
type Tile struct {
    Owner int
    Army  int
}

type Board struct {
    W, H int
    T    []Tile // length = W*H (row‑major)
}

func NewBoard(w, h int) *Board {
    b := &Board{W: w, H: h, T: make([]Tile, w*h)}
    // initialise all tiles as neutral
    for i := range b.T {
        b.T[i].Owner = -1
        b.T[i].Army = 0
    }
    return b
}

func (b *Board) Idx(x, y int) int { return y*b.W + x }