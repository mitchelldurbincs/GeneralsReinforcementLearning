package game

import (
    "fmt"
    "math/rand"
    "strings"
)

// Engine drives the game loop.
type Engine struct {
    gs *GameState
}

// NewEngine creates a WxH board with the given number of players and
// randomly places each player’s general (army=1) on an empty tile.
func NewEngine(w, h, players int) *Engine {
    b := NewBoard(w, h)

    gs := &GameState{
        Board:   b,
        Players: make([]Player, players),
    }

    // initialise players and place generals
    for pid := 0; pid < players; pid++ {
        gs.Players[pid] = Player{ID: pid, Alive: true}
        for {
            x := rand.Intn(w)
            y := rand.Intn(h)
            idx := b.Idx(x, y)
            if b.T[idx].Owner == -1 {
                b.T[idx].Owner = pid
                b.T[idx].Army = 1
                break
            }
        }
    }

    return &Engine{gs: gs}
}

func (e *Engine) Step() {
    e.gs.Turn++
    // simple growth: each owned tile gains +1 army
    for i := range e.gs.Board.T {
        if owner := e.gs.Board.T[i].Owner; owner != -1 {
            e.gs.Board.T[i].Army++
        }
    }
}

// Board returns an ASCII snapshot showing owner IDs (.) for neutral.
func (e *Engine) Board() string {
    var sb strings.Builder
    for y := 0; y < e.gs.Board.H; y++ {
        for x := 0; x < e.gs.Board.W; x++ {
            t := e.gs.Board.T[e.gs.Board.Idx(x, y)]
            if t.Owner == -1 {
                sb.WriteString(" . ")  // Neutral tile
            } else {
                // Show owner ID and army size
                sb.WriteString(fmt.Sprintf(" %d:%d ", t.Owner, t.Army))
            }
        }
        sb.WriteByte('\n')
    }
    return sb.String()
}