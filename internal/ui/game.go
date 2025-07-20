package ui

import (
	"context"
	"fmt"
	"image/color"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/config"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/ui/renderer"
)

// UI configuration functions
func ScreenWidth() int {
	return config.Get().UI.Window.Width
}

func ScreenHeight() int {
	return config.Get().UI.Window.Height
}

func TileSize() int {
	return config.Get().UI.Game.TileSize
}

func TurnInterval() int {
	return config.Get().UI.Game.TurnInterval
}

// UIGame holds the game engine instance and UI-specific state
type UIGame struct {
	engine        *game.Engine
	boardRenderer *renderer.BoardRenderer
	defaultFont   font.Face
	playerID      int

	// For simulation
	rng       *rand.Rand
	turnTimer int
}

// NewUIGame creates a new Ebitengine game instance.
func NewUIGame(engine *game.Engine, playerID int) (*UIGame, error) {
	g := &UIGame{
		engine:      engine,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
		turnTimer:   0,
		defaultFont: basicfont.Face7x13,
		playerID:    playerID,
	}

	g.boardRenderer = renderer.NewBoardRenderer(TileSize(), g.defaultFont)

	return g, nil
}

// Update proceeds the game state.
func (g *UIGame) Update() error {
	g.turnTimer++
	if g.turnTimer < TurnInterval() || g.engine.IsGameOver() {
		return nil
	}
	g.turnTimer = 0

	gs := g.engine.GameState()
	var actions []core.Action

	// On turn 0, there are no armies to move, so just step the engine
	if gs.Turn == 0 {
		g.engine.Step(context.Background(), nil)
		return nil
	}

	for _, player := range gs.Players {
		if !player.Alive {
			continue
		}

		var potentialSources []int
		for i, tile := range gs.Board.T {
			if tile.Owner == player.ID && tile.Army > 1 {
				potentialSources = append(potentialSources, i)
			}
		}

		if len(potentialSources) == 0 {
			continue
		}

		sourceIdx := potentialSources[g.rng.Intn(len(potentialSources))]
		fromX, fromY := gs.Board.XY(sourceIdx)

		directions := [][2]int{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}
		d := directions[g.rng.Intn(len(directions))]
		toX, toY := fromX+d[0], fromY+d[1]

		if toX >= 0 && toX < gs.Board.W && toY >= 0 && toY < gs.Board.H {
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

	// Always step the engine, even with no actions, to process production
	g.engine.Step(context.Background(), actions)

	return nil
}

// Draw renders the game screen.
func (g *UIGame) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{R: 50, G: 50, B: 50, A: 255}) // Dark gray background

	currentGameState := g.engine.GameState()

	if g.boardRenderer != nil {
		g.boardRenderer.Draw(screen, currentGameState.Board, currentGameState.Players, g.playerID)
	}

	turnStr := fmt.Sprintf("Turn: %d", currentGameState.Turn)
	ebitenutil.DebugPrintAt(screen, turnStr, 5, 5)

	for i, player := range currentGameState.Players {
		playerStr := fmt.Sprintf("Player %d: Army=%d, Alive=%t", player.ID, player.ArmyCount, player.Alive)
		ebitenutil.DebugPrintAt(screen, playerStr, 5, 25+i*20)
	}
}

// Layout defines the Ebitengine screen size.
func (g *UIGame) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return ScreenWidth(), ScreenHeight()
}
