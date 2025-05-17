package game

import "fmt"

// Engine drives the game loop.
type Engine struct {
    gs *GameState
}

func NewEngine(w, h, players int) *Engine {
    gs := &GameState{
        Board: NewBoard(w, h),
    }
    // TODO: spawn generals, armies, etc.
    return &Engine{gs: gs}
}

func (e *Engine) Step() {
    e.gs.Turn++
    // TODO: implement growth, moves, combat, win check
}

// Board returns a quick ASCII snapshot
func (e *Engine) Board() string {
    out := ""
    for y := 0; y < e.gs.Board.H; y++ {
        for x := 0; x < e.gs.Board.W; x++ {
            t := e.gs.Board.T[e.gs.Board.Idx(x, y)]
            out += fmt.Sprintf("%2d", t.Owner)
        }
        out += "\n"
    }
    return out
}