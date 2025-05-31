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

	var allCaptureDetailsThisTurn []core.CaptureDetails

	for _, action := range actions {
		playerID := action.GetPlayerID()
		// Validate player is alive (using the Players slice from GameState)
		if playerID < 0 || playerID >= len(e.gs.Players) || !e.gs.Players[playerID].Alive {
			continue // Skip actions from dead or invalid players
		}

		switch act := action.(type) {
		case *core.MoveAction:
			// ApplyMoveAction now returns (*CaptureDetails, error)
			captureDetail, err := core.ApplyMoveAction(e.gs.Board, act)
			if err != nil {
				// Log error (e.g., using a proper logger)
				fmt.Printf("Player %d action error: %v (Action: %+v)\n", act.PlayerID, err, act)
				// Depending on game rules, you might return err here or just skip this action
				continue
			}
			if captureDetail != nil {
				allCaptureDetailsThisTurn = append(allCaptureDetailsThisTurn, *captureDetail)
			}
		}
	}

	// Process captures to identify eliminations
	eliminationOrders := core.ProcessCaptures(allCaptureDetailsThisTurn)
	if len(eliminationOrders) > 0 {
		e.handleEliminationsAndTileTurnover(eliminationOrders)
	}

	return nil // Or return aggregated errors if any occurred and were not 'continue'd
}

// handleEliminationsAndTileTurnover processes player eliminations and transfers tiles.
func (e *Engine) handleEliminationsAndTileTurnover(orders []core.PlayerEliminationOrder) {
	for _, order := range orders {
		eliminatedID := order.EliminatedPlayerID
		newOwnerID := order.NewOwnerID

		fmt.Printf("Player %d's General captured by Player %d! Transferring assets.\n", eliminatedID, newOwnerID)

		// Iterate through all tiles on the board
		for i := range e.gs.Board.T {
			if e.gs.Board.T[i].Owner == eliminatedID {
				e.gs.Board.T[i].Owner = newOwnerID
				// Armies on the tiles remain as they are, as per your requirement.
				// If the tile was a city or the captured general itself, its type doesn't change here,
				// only its ownership. The captured general tile itself would have already been
				// flipped to newOwnerID by ApplyMoveAction. This loop ensures all *other*
				// tiles owned by eliminatedID are also flipped.
			}
		}
		// Note: The actual marking of the player as 'dead' (e.g. e.gs.Players[eliminatedID].Alive = false)
		// will be handled by `updatePlayerStats()` when it no longer finds a general for eliminatedID.
		// This keeps `updatePlayerStats` as the single source of truth for Alive status based on general presence.
	}
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
func (e *Engine) Board() string { // e.g. e is *game.Engine which has e.gs *game.State
	var sb strings.Builder

	// Unicode symbols for better visual distinction
	const (
		EmptySymbol    = "·"
		CitySymbol     = "⬢" // Hexagon
		GeneralSymbol  = "♔" // Crown
		MountainSymbol = "▲" // Triangle for mountain
		PlayerSymbols  = "ABCDEFGH"
	)

	// Column headers
	sb.WriteString("   ") // Adjusted for row numbers
	for x := 0; x < e.gs.Board.W; x++ {
		sb.WriteString(fmt.Sprintf("%2d", x))
	}
	sb.WriteString("\n")

	for y := 0; y < e.gs.Board.H; y++ {
		sb.WriteString(fmt.Sprintf("%2d ", y))

		for x := 0; x < e.gs.Board.W; x++ {
			// Assuming e.gs.Board.Tile(x,y) or similar method exists if T is not public
			// Or direct access if T is public and Idx method is on Board
			t := e.gs.Board.T[e.gs.Board.Idx(x, y)] // Using direct access as in original

			var symbol string
			var color string

			switch {
			case t.IsMountain(): // Priority 1: Mountains
				symbol = " " + MountainSymbol
				color = ColorGray // Mountains are gray
			
			case t.IsGeneral(): // Priority 2: Generals (owned)
				playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
				symbol = playerChar + GeneralSymbol
				color = getPlayerColor(t.Owner)

			case t.IsCity() && t.IsNeutral(): // Priority 3: Neutral Cities
				symbol = " " + CitySymbol
				color = ColorWhite // Neutral cities are white (or gray, depends on preference)
			
			case t.IsCity(): // Priority 4: Player-owned Cities
				playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
				symbol = playerChar + CitySymbol
				color = getPlayerColor(t.Owner)
			
			// case t.IsNeutral() && t.Army == 0 && t.Type == core.TileNormal: // More specific empty
			case t.IsNeutral() && t.Type == core.TileNormal: // Priority 5: Empty, neutral, normal land
				// If army > 0 on neutral normal, it will be handled by default or a new specific case
				if t.Army == 0 {
					symbol = " " + EmptySymbol
				} else {
					// Neutral land with armies (e.g. from a previous owner)
					symbol = fmt.Sprintf("%2d", t.Army) // Display army count for neutral land
					if t.Army >= 10 { // Keep it to 2 chars
						symbol = " +" // Or some other indicator for large neutral armies
					}
				}
				color = ColorGray


			default: // Player-owned normal tiles, or other neutral tiles with armies
				if t.IsNeutral() { // Neutral tile (not city, not mountain, not empty normal) e.g. with army
					if t.Army < 10 {
						symbol = fmt.Sprintf(" %d", t.Army)
					} else if t.Army < 100 { // Example for two digits
						symbol = fmt.Sprintf("%2d", t.Army)
					} else {
						symbol = "++" // For very large armies
					}
					color = ColorGray
				} else { // Player owned normal tile
					playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
					if t.Army < 10 {
						symbol = playerChar + fmt.Sprintf("%d", t.Army)
					} else {
						// For armies >= 10 on player tiles, use P+ (Player initial + plus sign)
						symbol = playerChar + "+"
					}
					color = getPlayerColor(t.Owner)
				}
			}
			sb.WriteString(color + symbol + ColorReset)
		}
		sb.WriteString("\n")
	}

	// Compact legend
	sb.WriteString("\n" + EmptySymbol + "=empty " + CitySymbol + "=city " + GeneralSymbol + "=general " + MountainSymbol + "=mountain A-H=players\n")

	return sb.String()
}

func getPlayerColor(playerID int) string {
	if playerID >= 0 && playerID < len(playerColors) {
		return playerColors[playerID]
	}
	return ColorWhite
}