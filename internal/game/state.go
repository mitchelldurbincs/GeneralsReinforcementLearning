package game

// Player holds per‑player state.
// For now we just track whether they are alive.
type Player struct {
    ID    int
    Alive bool
}

type GameState struct {
    Turn    int
    Board   *Board
    Players []Player
}