package game
import (
    "fmt"
    "math/rand"
    "strings"
)

// Constants for tile types
const (
    TileNormal  = 0
    TileGeneral = 1
    TileCity    = 2
)

// Engine drives the game loop.
type Engine struct {
    gs *GameState
}

// NewEngine creates a WxH board with the given number of players and
// randomly places each player's general (army=1) on an empty tile.
func NewEngine(w, h, players int) *Engine {
    b := NewBoard(w, h)
    gs := &GameState{
        Board:   b,
        Players: make([]Player, players),
    }
    
    // Place cities (optional - typically 1 city per every 7x7 area)
    // 5% of the map should be cities
    numCities := (w * h) / 20
    for i := 0; i < numCities; i++ {
        for {
            x := rand.Intn(w)
            y := rand.Intn(h)
            idx := b.Idx(x, y)
            if b.T[idx].Owner == -1 && b.T[idx].Type == TileNormal {
                b.T[idx].Type = TileCity
                b.T[idx].Army = 40 // Cities start with 40 armies in generals.io
                break
            }
        }
    }
    
    // Initialize players and place generals
    for pid := 0; pid < players; pid++ {
        gs.Players[pid] = Player{ID: pid, Alive: true}
        
        // Keep trying until we find a suitable location for the general
        for {
            x := rand.Intn(w)
            y := rand.Intn(h)
            idx := b.Idx(x, y)
            
            // Check if tile is empty and far enough from other generals
            if b.T[idx].Owner == -1 && b.T[idx].Type == TileNormal {
                // In generals.io, generals should be at least 5 tiles from other generals
                // Add distance check here if needed
                
                b.T[idx].Owner = pid
                b.T[idx].Army = 1
                b.T[idx].Type = TileGeneral
                break
            }
        }
    }
    
    return &Engine{gs: gs}
}

func (e *Engine) Step() {
    e.gs.Turn++
    
    // Generals produce 1 army every turn
    // Cities produce 1 army every turn if owned
    for i := range e.gs.Board.T {
        tile := &e.gs.Board.T[i]
        
        if tile.Owner != -1 {
            // Owned tile
            if tile.Type == TileGeneral || tile.Type == TileCity {
                // Generals and cities produce 1 army every turn
                tile.Army++
            } else if e.gs.Turn % 25 == 0 {
                // Normal tiles produce 1 army every 25 turns
                tile.Army++
            }
        }
    }
}

// Board returns an ASCII snapshot showing owner IDs and army sizes.
func (e *Engine) Board() string {
    var sb strings.Builder
    for y := 0; y < e.gs.Board.H; y++ {
        for x := 0; x < e.gs.Board.W; x++ {
            t := e.gs.Board.T[e.gs.Board.Idx(x, y)]
            
            if t.Owner == -1 {
                if t.Type == TileCity {
                    sb.WriteString(fmt.Sprintf(" C%d ", t.Army)) // City
                } else {
                    sb.WriteString(" .  ") // Neutral normal tile
                }
            } else {
                // Show owner ID, army size, and special indicator if needed
                if t.Type == TileGeneral {
                    sb.WriteString(fmt.Sprintf(" %d:%dG", t.Owner, t.Army)) // General
                } else if t.Type == TileCity {
                    sb.WriteString(fmt.Sprintf(" %d:%dC", t.Owner, t.Army)) // City
                } else {
                    sb.WriteString(fmt.Sprintf(" %d:%d ", t.Owner, t.Army)) // Normal
                }
            }
        }
        sb.WriteByte('\n')
    }
    return sb.String()
}