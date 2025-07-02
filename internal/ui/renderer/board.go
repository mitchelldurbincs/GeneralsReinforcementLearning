package renderer

import (
	"image/color"
	"strconv"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

// -----------------------------------------------------------------------------
// Colour definitions
// -----------------------------------------------------------------------------

var PlayerColors = map[int]color.Color{
	core.NeutralID: color.RGBA{120, 120, 120, 255}, // Neutral – gray
	0:              color.RGBA{200, 50, 50, 255},   // Red
	1:              color.RGBA{50, 100, 200, 255},  // Blue
	2:              color.RGBA{50, 200, 50, 255},   // Green
	3:              color.RGBA{200, 200, 50, 255},  // Yellow
}

var (
	MountainColor      = color.RGBA{80, 80, 80, 255}
	CityOwnedHueShift  = 30
	GeneralSymbolColor = color.White
	CitySymbolColor    = color.White
	ArmyTextColor      = color.White
)

// -----------------------------------------------------------------------------
// Renderer
// -----------------------------------------------------------------------------

type BoardRenderer struct {
	tileSize    int
	defaultFont font.Face
}

// NewBoardRenderer returns a renderer ready to use.
func NewBoardRenderer(tileSize int, f font.Face) *BoardRenderer {
	return &BoardRenderer{tileSize: tileSize, defaultFont: f}
}

// Draw renders the board on the supplied Ebiten screen.
func (br *BoardRenderer) Draw(screen *ebiten.Image, board *core.Board, players []game.Player) {
	if board == nil {
		return
	}

	for i, tile := range board.T {
		gridX, gridY := board.XY(i)

		screenX := float64(gridX * br.tileSize)
		screenY := float64(gridY * br.tileSize)

		// ---------------------------------------------------------------------
		// Choose base colour
		// ---------------------------------------------------------------------
		tileColor := PlayerColors[core.NeutralID]
		if c, ok := PlayerColors[tile.Owner]; ok {
			tileColor = c
		}

		cell := ebiten.NewImage(br.tileSize, br.tileSize)

		// ---------------------------------------------------------------------
		// Background pass
		// ---------------------------------------------------------------------
		switch {
		case tile.IsMountain():
			cell.Fill(MountainColor)

		default: // land / city / general
			cell.Fill(tileColor)

			// owned city (shaded inner square)
			if tile.IsCity() && tile.Owner != core.NeutralID {
				m := br.tileSize / 3
				sq := ebiten.NewImage(m, m)
				sq.Fill(shiftColor(tileColor, CityOwnedHueShift))
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(float64(br.tileSize-m)/2, float64(br.tileSize-m)/2)
				cell.DrawImage(sq, op)
			}

			// neutral city marker
			if tile.IsCity() && tile.Owner == core.NeutralID {
				m := br.tileSize / 2
				sq := ebiten.NewImage(m, m)
				sq.Fill(color.RGBA{180, 180, 180, 255})
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(float64(br.tileSize-m)/2, float64(br.tileSize-m)/2)
				cell.DrawImage(sq, op)
			}

			// general marker
			if tile.IsGeneral() {
				m := br.tileSize / 2
				sq := ebiten.NewImage(m, m)
				sq.Fill(GeneralSymbolColor)
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(float64(br.tileSize-m)/2, float64(br.tileSize-m)/2)
				cell.DrawImage(sq, op)
			}
		}

		// ---------------------------------------------------------------------
		// Blit cell to screen
		// ---------------------------------------------------------------------
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(screenX, screenY)
		screen.DrawImage(cell, op)

		// ---------------------------------------------------------------------
		// Army count (skip mountains & zero-armies)
		// ---------------------------------------------------------------------
		if tile.Army > 0 && !tile.IsMountain() && br.defaultFont != nil {
			armyStr := strconv.Itoa(tile.Army)

			// text bounds in pixels
			b := text.BoundString(br.defaultFont, armyStr)
			textW := b.Max.X - b.Min.X
			textH := b.Max.Y - b.Min.Y

			x := int(screenX) + (br.tileSize-textW)/2
			y := int(screenY) + (br.tileSize+textH)/2

			text.Draw(screen, armyStr, br.defaultFont, x, y, ArmyTextColor)
		}
	}
}

// shiftColor returns a slightly lighter version of c.
func shiftColor(c color.Color, amount int) color.Color {
	r, g, b, a := c.RGBA()
	inc := uint32(amount) << 8 // amount*256

	r = clamp16(r + inc)
	g = clamp16(g + inc)
	b = clamp16(b + inc)
	return color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
}

func clamp16(v uint32) uint32 {
	const max = 0xFFFF
	if v > max {
		return max
	}
	return v
}
