package ui

import (
	"context"
	"fmt"
	"image/color"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/rs/zerolog/log"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/common"
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
	currentTurnPlayer int
	
	// Game progression
	framesSinceStep int
	stepsPerSecond  int // How many game steps per second
	accumulatedActions map[int][]core.Action // Actions accumulated per player
	
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
		defaultFont:   basicfont.Face7x13,
		playerConfigs: playerConfigs,
		humanPlayerID: humanPlayerID,
		currentTurnPlayer: 0,
		framesSinceStep: 0,
		stepsPerSecond: 2, // Two game steps per second for comfortable gameplay
		accumulatedActions: make(map[int][]core.Action),
	}
	
	g.boardRenderer = renderer.NewEnhancedBoardRenderer(TileSize(), g.defaultFont)
	g.inputHandler = input.NewHandler(TileSize())
	g.inputHandler.SetBoardOffset(0, 0) // Board is rendered at origin
	
	// Set up tile validator
	g.inputHandler.SetTileValidator(func(x, y int) (bool, string) {
		gs := g.engine.GameState()
		if !gs.Board.InBounds(x, y) {
			return false, "Out of bounds"
		}
		
		idx := gs.Board.Idx(x, y)
		tile := gs.Board.T[idx]
		
		// Check visibility if fog of war is enabled
		if gs.FogOfWarEnabled && !tile.IsVisibleTo(g.humanPlayerID) {
			return false, "Tile not visible"
		}
		
		// Check if tile belongs to the human player
		if tile.Owner != g.humanPlayerID {
			return false, "Not your tile"
		}
		
		// Check if tile has enough army
		if tile.Army <= 1 {
			return false, "Not enough army"
		}
		
		return true, ""
	})
	
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
	
	// Handle continuous input from human player
	if g.humanPlayerID >= 0 && g.playerConfigs[g.humanPlayerID].Type == PlayerTypeHuman {
		g.handleHumanInput()
	}
	
	// Handle AI players continuously
	g.handleAIPlayers()
	
	// Automatic game progression
	g.framesSinceStep++
	framesPerStep := 60 / g.stepsPerSecond // 60 FPS assumed
	
	if g.framesSinceStep >= framesPerStep {
		g.framesSinceStep = 0
		
		// Collect all accumulated actions
		allActions := []core.Action{}
		for playerID, actions := range g.accumulatedActions {
			allActions = append(allActions, actions...)
			if len(actions) > 0 {
				log.Debug().
					Int("playerID", playerID).
					Int("actionCount", len(actions)).
					Msg("Submitting player actions")
			}
		}
		
		// Clear accumulated actions
		g.accumulatedActions = make(map[int][]core.Action)
		
		// Step the game
		if len(allActions) > 0 {
			log.Debug().Int("totalActions", len(allActions)).Msg("Stepping game with actions")
		}
		g.engine.Step(context.Background(), allActions)
		
		// Update turn counter display
		g.currentTurnPlayer = (g.currentTurnPlayer + 1) % len(g.playerConfigs)
	}
	
	return nil
}

func (g *HumanGame) handleHumanInput() {
	g.inputHandler.SetPlayerTurn(true)
	
	gs := g.engine.GameState()
	
	// Check for validation messages
	if msg := g.inputHandler.GetLastValidationMessage(); msg != "" {
		g.showMessage(msg, 60)
	}
	
	// Check for tile selection and movement
	pendingMoves := g.inputHandler.GetPendingMoves()
	
	if len(pendingMoves) > 0 {
		log.Debug().Int("count", len(pendingMoves)).Msg("Processing pending moves")
	}
	
	for _, move := range pendingMoves {
		log.Debug().
			Int("fromX", move.FromX).Int("fromY", move.FromY).
			Int("toX", move.ToX).Int("toY", move.ToY).
			Bool("moveHalf", move.MoveHalf).
			Msg("Processing move")
		// Validate the move
		fromIdx := gs.Board.Idx(move.FromX, move.FromY)
		tile := gs.Board.T[fromIdx]
		
		if tile.Owner != g.humanPlayerID || tile.Army <= 1 {
			g.showMessage("Invalid source tile", 60)
			continue
		}
		
		// Check if destination is adjacent
		dx := common.Abs(move.ToX - move.FromX)
		dy := common.Abs(move.ToY - move.FromY)
		if dx+dy != 1 {
			g.showMessage("Can only move to adjacent tiles", 60)
			continue
		}
		
		// Check if destination is in bounds
		if !gs.Board.InBounds(move.ToX, move.ToY) {
			continue
		}
		
		// Check if destination is not a mountain
		toIdx := gs.Board.Idx(move.ToX, move.ToY)
		toTile := gs.Board.T[toIdx]
		
		// Check visibility of destination if fog of war is enabled
		if gs.FogOfWarEnabled && !toTile.IsVisibleTo(g.humanPlayerID) {
			g.showMessage("Cannot move to unseen tiles", 60)
			continue
		}
		
		if toTile.IsMountain() {
			g.showMessage("Cannot move to mountains", 60)
			continue
		}
		
		action := &core.MoveAction{
			PlayerID: g.humanPlayerID,
			FromX:    move.FromX,
			FromY:    move.FromY,
			ToX:      move.ToX,
			ToY:      move.ToY,
			MoveAll:  !move.MoveHalf,
		}
		
		// Add to accumulated actions
		g.accumulatedActions[g.humanPlayerID] = append(g.accumulatedActions[g.humanPlayerID], action)
		log.Debug().
			Int("playerID", g.humanPlayerID).
			Int("totalActions", len(g.accumulatedActions[g.humanPlayerID])).
			Msg("Added move action to accumulated actions")
	}
	
	g.inputHandler.ClearPendingMoves()
}

func (g *HumanGame) handleAIPlayers() {
	gs := g.engine.GameState()
	
	// Process each AI player
	for i, config := range g.playerConfigs {
		if config.Type != PlayerTypeAI {
			continue
		}
		
		player := gs.Players[i]
		if !player.Alive {
			continue
		}
		
		// Check if enough time has passed for this AI to make a move
		// Each AI gets its own timer to avoid all AIs moving at once
		if g.turnTimer%30 != i*10 { // Stagger AI moves
			continue
		}
		
		// Simple random AI - make one move per step
		var potentialSources []int
		for idx, tile := range gs.Board.T {
			if tile.Owner == player.ID && tile.Army > 1 {
				potentialSources = append(potentialSources, idx)
			}
		}
		
		if len(potentialSources) > 0 {
			sourceIdx := potentialSources[g.rng.Intn(len(potentialSources))]
			fromX, fromY := gs.Board.XY(sourceIdx)
			
			directions := [][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}
			d := directions[g.rng.Intn(len(directions))]
			toX, toY := fromX+d[0], fromY+d[1]
			
			if gs.Board.InBounds(toX, toY) {
				toIdx := gs.Board.Idx(toX, toY)
				if !gs.Board.T[toIdx].IsMountain() {
					action := &core.MoveAction{
						PlayerID: player.ID,
						FromX:    fromX,
						FromY:    fromY,
						ToX:      toX,
						ToY:      toY,
						MoveAll:  g.rng.Intn(2) == 0,
					}
					g.accumulatedActions[player.ID] = append(g.accumulatedActions[player.ID], action)
				}
			}
		}
	}
	
	g.turnTimer++
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
	g.drawUI(screen, &currentGameState)
}

func (g *HumanGame) drawUI(screen *ebiten.Image, gs *game.GameState) {
	// Turn and player info
	turnStr := fmt.Sprintf("Turn: %d", gs.Turn)
	ebitenutil.DebugPrintAt(screen, turnStr, 5, 5)
	
	// Game is now continuous, show next step timer
	timeToNextStep := float64(60/g.stepsPerSecond - g.framesSinceStep) / 60.0
	stepStr := fmt.Sprintf("Next step in: %.1fs", timeToNextStep)
	text.Draw(screen, stepStr, g.defaultFont, 5, 25, color.White)
	
	// Player stats
	for i, player := range gs.Players {
		y := 45 + i*20
		playerStr := fmt.Sprintf("P%d: Army=%d", player.ID, player.ArmyCount)
		if !player.Alive {
			playerStr += " (Dead)"
		}
		
		// Use player color for the text
		playerColor := common.PlayerColors[player.ID]
		text.Draw(screen, playerStr, g.defaultFont, 5, y, playerColor)
	}
	
	// Controls help
	helpY := ScreenHeight() - 80
	text.Draw(screen, "Controls:", g.defaultFont, 5, helpY, color.White)
	text.Draw(screen, "Click: Select/Move", g.defaultFont, 5, helpY+15, color.Gray{200})
	text.Draw(screen, "Q/W: Full/Half army", g.defaultFont, 5, helpY+30, color.Gray{200})
	text.Draw(screen, "ESC: Deselect", g.defaultFont, 5, helpY+45, color.Gray{200})
	
	// Move mode indicator
	modeStr := "Move Mode: "
	if g.inputHandler.GetMoveMode() == input.MoveFull {
		modeStr += "Full Army"
	} else {
		modeStr += "Half Army"
	}
	text.Draw(screen, modeStr, g.defaultFont, ScreenWidth()-150, 5, color.White)
	
	// Status message
	if g.messageTimer > 0 && g.statusMessage != "" {
		msgX := ScreenWidth()/2 - len(g.statusMessage)*3
		msgY := ScreenHeight() - 20
		text.Draw(screen, g.statusMessage, g.defaultFont, msgX, msgY, color.White)
	}
}

func (g *HumanGame) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return ScreenWidth(), ScreenHeight()
}

