package renderer

import (
	"image/color"
	"strconv"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/common"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
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
func (br *BoardRenderer) Draw(screen *ebiten.Image, board *core.Board, players []game.Player, playerID int) {
	if board == nil {
		return
	}

	for i, tile := range board.T {
		gridX, gridY := board.XY(i)

		screenX := float64(gridX * br.tileSize)
		screenY := float64(gridY * br.tileSize)

		// ---------------------------------------------------------------------
		visible := playerID < 0 || tile.IsVisibleTo(playerID)

		cell := ebiten.NewImage(br.tileSize, br.tileSize)

		// ---------------------------------------------------------------------
		// Background pass
		// ---------------------------------------------------------------------
		switch {
		case tile.IsMountain():
			cell.Fill(common.MountainColor)

		default: // land / city / general
			// Only show player colors for visible tiles
			tileColor := common.PlayerColors[core.NeutralID]
			if visible {
				// For visible tiles, show the actual owner's color
				if c, ok := common.PlayerColors[tile.Owner]; ok {
					tileColor = c
				}
			}
			// For non-visible tiles, we use the neutral gray color
			cell.Fill(tileColor)

			// owned city (shaded inner square)
			if visible && tile.IsCity() && tile.Owner != core.NeutralID {
				m := br.tileSize / 3
				sq := ebiten.NewImage(m, m)
				sq.Fill(shiftColor(tileColor, common.CityOwnedHueShift))
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(float64(br.tileSize-m)/2, float64(br.tileSize-m)/2)
				cell.DrawImage(sq, op)
			}

			// neutral city marker
			if tile.IsCity() && (visible || tile.Owner == core.NeutralID) {
				m := br.tileSize / 2
				sq := ebiten.NewImage(m, m)
				sq.Fill(color.RGBA{180, 180, 180, 255})
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(float64(br.tileSize-m)/2, float64(br.tileSize-m)/2)
				cell.DrawImage(sq, op)
			}

			// general marker
			if visible && tile.IsGeneral() {
				m := br.tileSize / 2
				sq := ebiten.NewImage(m, m)
				sq.Fill(common.GeneralSymbolColor)
				op := &ebiten.DrawImageOptions{}
				op.GeoM.Translate(float64(br.tileSize-m)/2, float64(br.tileSize-m)/2)
				cell.DrawImage(sq, op)
			}
		}

		if !visible {
			fog := ebiten.NewImage(br.tileSize, br.tileSize)
			fog.Fill(color.RGBA{25, 25, 25, 200}) // More opaque fog
			cell.DrawImage(fog, nil)
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
		if visible && tile.Army > 0 && !tile.IsMountain() && br.defaultFont != nil {
			armyStr := strconv.Itoa(tile.Army)

			textColor := common.ArmyTextColor
			if tile.IsGeneral() {
				textColor = common.GeneralArmyTextColor
			}

			// text bounds in pixels
			b := text.BoundString(br.defaultFont, armyStr)
			textW := b.Max.X - b.Min.X
			textH := b.Max.Y - b.Min.Y

			x := int(screenX) + (br.tileSize-textW)/2
			y := int(screenY) + (br.tileSize+textH)/2

			text.Draw(screen, armyStr, br.defaultFont, x, y, textColor)
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
