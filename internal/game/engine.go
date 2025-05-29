package game

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/mapgen"
)

type Engine struct {
	gs       *GameState
	rng      *rand.Rand
	gameOver bool
}

// NewEngine creates a new game engine with map generation
func NewEngine(w, h, players int, rng *rand.Rand) *Engine {
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Use mapgen package
	config := mapgen.DefaultMapConfig(w, h, players)
	generator := mapgen.NewGenerator(config, rng)
	board := generator.GenerateMap()

	// Initialize players
	playerSlice := make([]Player, players)
	for i := 0; i < players; i++ {
		playerSlice[i] = Player{
			ID:         i,
			Alive:      true,
			GeneralIdx: -1, // Will be set by updatePlayerStats
		}
	}

	e := &Engine{
		gs: &GameState{
			Board:   board,
			Players: playerSlice,
		},
		rng:      rng,
		gameOver: false,
	}

	e.updatePlayerStats()
	return e
}

// Step processes actions and advances the game by one turn
func (e *Engine) Step(actions []core.Action) error {
	if e.gameOver {
		return core.ErrGameOver
	}

	e.gs.Turn++

	// Process actions
	if err := e.processActions(actions); err != nil {
		return err
	}

	// Apply production
	e.processTurnProduction()

	// Update player stats and check win conditions
	e.updatePlayerStats()
	e.checkGameOver()

	return nil
}

// processActions handles all actions for this turn
func (e *Engine) processActions(actions []core.Action) error {
	// Sort actions for deterministic processing (by player ID)
	sort.Slice(actions, func(i, j int) bool {
		return actions[i].GetPlayerID() < actions[j].GetPlayerID()
	})

	var captures []core.CaptureInfo

	for _, action := range actions {
		// Validate player is alive
		if !e.gs.Players[action.GetPlayerID()].Alive {
			continue // Skip actions from dead players
		}

		switch act := action.(type) {
		case *core.MoveAction:
			captured, err := core.ApplyMoveAction(e.gs.Board, act)
			if err != nil {
				// Log error but continue processing other actions
				// In a real implementation, you might want structured logging here
				continue
			}
			if captured {
				captures = append(captures, core.CaptureInfo{
					X:        act.ToX,
					Y:        act.ToY,
					PlayerID: act.PlayerID,
					TileType: e.gs.Board.T[e.gs.Board.Idx(act.ToX, act.ToY)].Type,
				})
			}
		}
	}

	// Process any special capture effects
	core.ProcessCaptures(e.gs.Board, captures)

	return nil
}

// processTurnProduction applies army growth
func (e *Engine) processTurnProduction() {
	growNormal := e.gs.Turn%NormalGrowInterval == 0

	for i := range e.gs.Board.T {
		t := &e.gs.Board.T[i]
		if t.IsNeutral() {
			continue
		}

		switch t.Type {
		case core.TileGeneral:
			t.Army += GeneralProduction
		case core.TileCity:
			t.Army += CityProduction
		case core.TileNormal:
			if growNormal {
				t.Army += NormalProduction
			}
		}
	}
}

// updatePlayerStats recalculates player statistics
func (e *Engine) updatePlayerStats() {
	// Reset stats
	for pid := range e.gs.Players {
		e.gs.Players[pid].ArmyCount = 0
		e.gs.Players[pid].GeneralIdx = -1
	}

	// Recalculate from board state
	for i, t := range e.gs.Board.T {
		if t.IsNeutral() {
			continue
		}

		p := &e.gs.Players[t.Owner]
		p.ArmyCount += t.Army

		if t.IsGeneral() {
			p.GeneralIdx = i
		}
	}

	// Update alive status
	for pid := range e.gs.Players {
		e.gs.Players[pid].Alive = e.gs.Players[pid].GeneralIdx != -1
	}
}

// checkGameOver determines if the game has ended
func (e *Engine) checkGameOver() {
	aliveCount := 0
	for _, p := range e.gs.Players {
		if p.Alive {
			aliveCount++
		}
	}
	e.gameOver = aliveCount <= 1
}

// Public accessors
func (e *Engine) GameState() GameState { return *e.gs }
func (e *Engine) IsGameOver() bool     { return e.gameOver }

// GetWinner returns the winning player ID, or -1 if game isn't over
func (e *Engine) GetWinner() int {
	if !e.gameOver {
		return -1
	}
	for _, p := range e.gs.Players {
		if p.Alive {
			return p.ID
		}
	}
	return -1 // Draw or no winner
}

// ANSI color codes
const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m" 
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
	ColorWhite  = "\033[37m"
	ColorGray   = "\033[90m"
	
	// Background colors for better visibility
	BgRed    = "\033[41m"
	BgGreen  = "\033[42m"
	BgYellow = "\033[43m"
	BgBlue   = "\033[44m"
	BgPurple = "\033[45m"
	BgCyan   = "\033[46m"
)

var playerColors = []string{ColorRed, ColorBlue, ColorGreen, ColorYellow, ColorPurple, ColorCyan}

// Alternative: Compact board with unicode symbols
func (e *Engine) Board() string {
	var sb strings.Builder
	
	// Unicode symbols for better visual distinction
	const (
		EmptySymbol   = "·"
		CitySymbol    = "⬢"
		GeneralSymbol = "♔"
		PlayerSymbols = "ABCDEFGH"
	)
	
	// Column headers
	sb.WriteString("   ")
	for x := range e.gs.Board.W {
		sb.WriteString(fmt.Sprintf("%2d", x))
	}
	sb.WriteString("\n")
	
	for y := range e.gs.Board.H {
		sb.WriteString(fmt.Sprintf("%2d ", y))
		
		for x := 0; x < e.gs.Board.W; x++ {
			t := e.gs.Board.T[e.gs.Board.Idx(x, y)]
			
			var symbol string
			var color string
			
			switch {
			case t.IsNeutral() && t.Army == 0:
				symbol = " " + EmptySymbol
				color = ColorGray
				
			case t.IsCity() && t.IsNeutral():
				symbol = " " + CitySymbol
				color = ColorWhite
				
			case t.IsGeneral():
				playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
				symbol = playerChar + GeneralSymbol
				color = getPlayerColor(t.Owner)
				
			case t.IsCity():
				playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
				symbol = playerChar + CitySymbol
				color = getPlayerColor(t.Owner)
				
			default:
				playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
				if t.Army < 10 {
					symbol = playerChar + fmt.Sprintf("%d", t.Army)
				} else {
					symbol = playerChar + "+"
				}
				color = getPlayerColor(t.Owner)
			}
			
			sb.WriteString(color + symbol + ColorReset)
		}
		sb.WriteString("\n")
	}
	
	// Compact legend
	sb.WriteString("\n" + EmptySymbol + "=empty " + CitySymbol + "=city " + GeneralSymbol + "=general A-H=players\n")
	
	return sb.String()
}

func getPlayerColor(playerID int) string {
	if playerID >= 0 && playerID < len(playerColors) {
		return playerColors[playerID]
	}
	return ColorWhite
}