package ui

import (
	"context"
	"fmt"
	"image/color"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/ui/input"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/ui/renderer"
)

type PlayerType int

const (
	PlayerTypeHuman PlayerType = iota
	PlayerTypeAI
)

type PlayerConfig struct {
	Type PlayerType
	ID   int
}

type HumanGame struct {
	engine        *game.Engine
	boardRenderer *renderer.EnhancedBoardRenderer
	inputHandler  *input.Handler
	defaultFont   font.Face
	
	// Player configuration
	playerConfigs []PlayerConfig
	humanPlayerID int
	
	// Turn management
	rng            *rand.Rand
	turnTimer      int
	autoTurnDelay  int // Frames to wait between AI turns
	waitingForHuman bool
	currentTurnPlayer int
	
	// UI state
	statusMessage  string
	messageTimer   int
}

func NewHumanGame(engine *game.Engine, playerConfigs []PlayerConfig) (*HumanGame, error) {
	// Find the first human player ID
	humanPlayerID := -1
	for _, config := range playerConfigs {
		if config.Type == PlayerTypeHuman {
			humanPlayerID = config.ID
			break
		}
	}
	
	g := &HumanGame{
		engine:        engine,
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())),
		turnTimer:     0,
		autoTurnDelay: 30, // 0.5 seconds at 60 FPS
		defaultFont:   basicfont.Face7x13,
		playerConfigs: playerConfigs,
		humanPlayerID: humanPlayerID,
		currentTurnPlayer: 0,
	}
	
	g.boardRenderer = renderer.NewEnhancedBoardRenderer(TileSize, g.defaultFont)
	g.inputHandler = input.NewHandler(TileSize)
	
	return g, nil
}

func (g *HumanGame) Update() error {
	// Update input handler
	g.inputHandler.Update()
	
	// Update message timer
	if g.messageTimer > 0 {
		g.messageTimer--
	}
	
	// Update board renderer with hover position
	hoverX, hoverY := g.inputHandler.GetHoveredTile()
	g.boardRenderer.SetHover(hoverX, hoverY)
	
	// Update selection state
	if selX, selY, hasSel := g.inputHandler.GetSelectedTile(); hasSel {
		g.boardRenderer.SetSelection(selX, selY, true)
	} else {
		g.boardRenderer.SetSelection(0, 0, false)
	}
	
	if g.engine.IsGameOver() {
		return nil
	}
	
	gs := g.engine.GameState()
	
	// Determine whose turn it is
	currentPlayer := gs.Players[g.currentTurnPlayer]
	if !currentPlayer.Alive {
		g.nextPlayer()
		return nil
	}
	
	currentConfig := g.playerConfigs[g.currentTurnPlayer]
	
	if currentConfig.Type == PlayerTypeHuman {
		g.handleHumanTurn(currentPlayer)
	} else {
		g.handleAITurn(currentPlayer)
	}
	
	return nil
}

func (g *HumanGame) handleHumanTurn(player game.Player) {
	g.inputHandler.SetPlayerTurn(true)
	g.waitingForHuman = true
	
	gs := g.engine.GameState()
	
	// Check for tile selection and movement
	pendingMoves := g.inputHandler.GetPendingMoves()
	validActions := []core.Action{}
	
	for _, move := range pendingMoves {
		// Validate the move
		fromIdx := gs.Board.I(move.FromX, move.FromY)
		tile := gs.Board.T[fromIdx]
		
		if tile.Owner != player.ID || tile.Army <= 1 {
			g.showMessage("Invalid source tile", 60)
			continue
		}
		
		// Check if destination is adjacent
		dx := abs(move.ToX - move.FromX)
		dy := abs(move.ToY - move.FromY)
		if dx+dy != 1 {
			g.showMessage("Can only move to adjacent tiles", 60)
			continue
		}
		
		// Check if destination is in bounds
		if !gs.Board.InBounds(move.ToX, move.ToY) {
			continue
		}
		
		// Check if destination is not a mountain
		toIdx := gs.Board.I(move.ToX, move.ToY)
		if gs.Board.T[toIdx].IsMountain() {
			g.showMessage("Cannot move to mountains", 60)
			continue
		}
		
		action := &core.MoveAction{
			PlayerID: player.ID,
			FromX:    move.FromX,
			FromY:    move.FromY,
			ToX:      move.ToX,
			ToY:      move.ToY,
			MoveAll:  !move.MoveHalf,
		}
		
		validActions = append(validActions, action)
	}
	
	g.inputHandler.ClearPendingMoves()
	
	// Execute moves or check for turn end
	if len(validActions) > 0 || g.inputHandler.IsTurnEnded() {
		g.engine.Step(context.Background(), validActions)
		g.waitingForHuman = false
		g.inputHandler.SetPlayerTurn(false)
		g.nextPlayer()
		
		if len(validActions) > 0 {
			g.showMessage(fmt.Sprintf("Executed %d moves", len(validActions)), 60)
		}
	}
}

func (g *HumanGame) handleAITurn(player game.Player) {
	g.turnTimer++
	if g.turnTimer < g.autoTurnDelay {
		return
	}
	g.turnTimer = 0
	
	gs := g.engine.GameState()
	var actions []core.Action
	
	// Simple random AI
	var potentialSources []int
	for i, tile := range gs.Board.T {
		if tile.Owner == player.ID && tile.Army > 1 {
			potentialSources = append(potentialSources, i)
		}
	}
	
	if len(potentialSources) > 0 {
		// Make 1-3 random moves
		numMoves := 1 + g.rng.Intn(3)
		for i := 0; i < numMoves && len(potentialSources) > 0; i++ {
			sourceIdx := potentialSources[g.rng.Intn(len(potentialSources))]
			fromX, fromY := gs.Board.XY(sourceIdx)
			
			directions := [][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}
			d := directions[g.rng.Intn(len(directions))]
			toX, toY := fromX+d[0], fromY+d[1]
			
			if gs.Board.InBounds(toX, toY) {
				toIdx := gs.Board.I(toX, toY)
				if !gs.Board.T[toIdx].IsMountain() {
					actions = append(actions, &core.MoveAction{
						PlayerID: player.ID,
						FromX:    fromX,
						FromY:    fromY,
						ToX:      toX,
						ToY:      toY,
						MoveAll:  g.rng.Intn(2) == 0,
					})
				}
			}
		}
	}
	
	g.engine.Step(context.Background(), actions)
	g.nextPlayer()
}

func (g *HumanGame) nextPlayer() {
	g.currentTurnPlayer = (g.currentTurnPlayer + 1) % len(g.playerConfigs)
	g.turnTimer = 0
}

func (g *HumanGame) showMessage(msg string, duration int) {
	g.statusMessage = msg
	g.messageTimer = duration
}

func (g *HumanGame) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{R: 50, G: 50, B: 50, A: 255})
	
	currentGameState := g.engine.GameState()
	
	// Draw the board with enhancements
	if g.boardRenderer != nil {
		g.boardRenderer.Draw(screen, currentGameState.Board, currentGameState.Players, g.humanPlayerID)
	}
	
	// Draw UI elements
	g.drawUI(screen, currentGameState)
}

func (g *HumanGame) drawUI(screen *ebiten.Image, gs *game.GameState) {
	// Turn and player info
	turnStr := fmt.Sprintf("Turn: %d", gs.Turn)
	ebitenutil.DebugPrintAt(screen, turnStr, 5, 5)
	
	// Current player indicator
	currentPlayer := gs.Players[g.currentTurnPlayer]
	currentStr := fmt.Sprintf("Current Turn: Player %d", currentPlayer.ID)
	if g.playerConfigs[g.currentTurnPlayer].Type == PlayerTypeHuman {
		currentStr += " (Human)"
	} else {
		currentStr += " (AI)"
	}
	text.Draw(screen, currentStr, g.defaultFont, 5, 25, color.White)
	
	// Player stats
	for i, player := range gs.Players {
		y := 45 + i*20
		playerStr := fmt.Sprintf("P%d: Army=%d", player.ID, player.ArmyCount)
		if !player.Alive {
			playerStr += " (Dead)"
		}
		
		// Use player color for the text
		playerColor := renderer.PlayerColors[player.ID]
		text.Draw(screen, playerStr, g.defaultFont, 5, y, playerColor)
	}
	
	// Controls help
	if g.waitingForHuman {
		helpY := ScreenHeight - 80
		text.Draw(screen, "Controls:", g.defaultFont, 5, helpY, color.White)
		text.Draw(screen, "Click: Select/Move", g.defaultFont, 5, helpY+15, color.Gray{200})
		text.Draw(screen, "Q/W: Full/Half army", g.defaultFont, 5, helpY+30, color.Gray{200})
		text.Draw(screen, "Space: End turn", g.defaultFont, 5, helpY+45, color.Gray{200})
		text.Draw(screen, "ESC: Deselect", g.defaultFont, 5, helpY+60, color.Gray{200})
		
		// Move mode indicator
		modeStr := "Move Mode: "
		if g.inputHandler.GetMoveMode() == input.MoveFull {
			modeStr += "Full Army"
		} else {
			modeStr += "Half Army"
		}
		text.Draw(screen, modeStr, g.defaultFont, ScreenWidth-150, 5, color.White)
	}
	
	// Status message
	if g.messageTimer > 0 && g.statusMessage != "" {
		msgX := ScreenWidth/2 - len(g.statusMessage)*3
		msgY := ScreenHeight - 20
		text.Draw(screen, g.statusMessage, g.defaultFont, msgX, msgY, color.White)
	}
}

func (g *HumanGame) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return ScreenWidth, ScreenHeight
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}