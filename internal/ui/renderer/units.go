package renderer

import (
	"image/color"
	"strconv"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font"
)

// ArmyRenderer handles drawing army counts with visual effects
type ArmyRenderer struct {
	defaultFont font.Face
}

// NewArmyRenderer creates a new army renderer
func NewArmyRenderer(f font.Face) *ArmyRenderer {
	return &ArmyRenderer{defaultFont: f}
}

// DrawArmyCount draws an army count with outline effect for better visibility
func (ar *ArmyRenderer) DrawArmyCount(screen *ebiten.Image, x, y int, army int, isGeneral bool, tileSize int) {
	if ar.defaultFont == nil || army <= 0 {
		return
	}

	armyStr := strconv.Itoa(army)

	// Determine colors based on tile type
	var textColor, outlineColor color.Color
	if isGeneral {
		textColor = color.Black
		outlineColor = color.RGBA{255, 255, 255, 200} // Light outline for dark text
	} else {
		textColor = color.White
		outlineColor = color.RGBA{0, 0, 0, 200} // Dark outline for light text
	}

	// Calculate centered text position
	b := text.BoundString(ar.defaultFont, armyStr)
	textW := b.Max.X - b.Min.X
	textH := b.Max.Y - b.Min.Y

	textX := x + (tileSize-textW)/2
	textY := y + (tileSize+textH)/2

	// Draw outline by drawing text at 4 diagonal offsets
	offsets := []struct{ dx, dy int }{
		{-1, -1}, {1, -1}, {-1, 1}, {1, 1},
	}
	for _, o := range offsets {
		text.Draw(screen, armyStr, ar.defaultFont, textX+o.dx, textY+o.dy, outlineColor)
	}

	// Add extra glow for large armies (100+)
	if army >= 100 {
		glowColor := color.RGBA{0, 0, 0, 100}
		if isGeneral {
			glowColor = color.RGBA{255, 255, 255, 100}
		}
		// Additional outer glow at distance 2
		outerOffsets := []struct{ dx, dy int }{
			{-2, 0}, {2, 0}, {0, -2}, {0, 2},
		}
		for _, o := range outerOffsets {
			text.Draw(screen, armyStr, ar.defaultFont, textX+o.dx, textY+o.dy, glowColor)
		}
	}

	// Draw main text on top
	text.Draw(screen, armyStr, ar.defaultFont, textX, textY, textColor)
}
