package renderer

import (
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/common"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

// FogRenderer handles fog of war visual effects including edge highlighting.
type FogRenderer struct {
	tileSize      int
	edgeThickness float32
}

// NewFogRenderer creates a new fog renderer for the given tile size.
func NewFogRenderer(tileSize int) *FogRenderer {
	return &FogRenderer{
		tileSize:      tileSize,
		edgeThickness: 2.0,
	}
}

// Direction represents a cardinal direction for edge detection
type direction struct {
	dx, dy int
	// Edge position relative to tile (0=top, 1=right, 2=bottom, 3=left)
	edge int
}

var cardinalDirs = []direction{
	{0, -1, 0}, // Up -> top edge
	{1, 0, 1},  // Right -> right edge
	{0, 1, 2},  // Down -> bottom edge
	{-1, 0, 3}, // Left -> left edge
}

// DrawEdgeHighlights renders visibility boundary highlights on the screen.
// It draws bright edges where visible tiles meet non-visible tiles.
func (fr *FogRenderer) DrawEdgeHighlights(screen *ebiten.Image, board *core.Board, playerID int) {
	if board == nil || playerID < 0 {
		return
	}

	size := float32(fr.tileSize)
	edgeColor := common.VisibilityEdgeColor

	for i := range board.T {
		tile := &board.T[i]

		// Only process visible tiles
		if !tile.IsVisibleTo(playerID) {
			continue
		}

		x, y := board.XY(i)
		screenX := float32(x) * size
		screenY := float32(y) * size

		// Check each cardinal direction for fog/shroud adjacency
		for _, dir := range cardinalDirs {
			adjX, adjY := x+dir.dx, y+dir.dy

			// Check if adjacent tile is not visible (either out of bounds or fogged/shrouded)
			isEdge := false
			if !board.InBounds(adjX, adjY) {
				// Board edge is also a visibility edge
				isEdge = true
			} else {
				adjTile := board.GetTile(adjX, adjY)
				if adjTile != nil && !adjTile.IsVisibleTo(playerID) {
					isEdge = true
				}
			}

			if isEdge {
				fr.drawEdgeLine(screen, screenX, screenY, size, dir.edge, edgeColor)
			}
		}
	}
}

// drawEdgeLine draws a single edge line on a tile.
// edge: 0=top, 1=right, 2=bottom, 3=left
func (fr *FogRenderer) drawEdgeLine(screen *ebiten.Image, screenX, screenY, size float32, edge int, clr interface{}) {
	t := fr.edgeThickness

	var x1, y1, x2, y2 float32

	switch edge {
	case 0: // Top edge
		x1, y1 = screenX, screenY
		x2, y2 = screenX+size, screenY
	case 1: // Right edge
		x1, y1 = screenX+size-t, screenY
		x2, y2 = screenX+size-t, screenY+size
	case 2: // Bottom edge
		x1, y1 = screenX, screenY+size-t
		x2, y2 = screenX+size, screenY+size-t
	case 3: // Left edge
		x1, y1 = screenX, screenY
		x2, y2 = screenX, screenY+size
	}

	// Draw as a thick line using vector graphics
	vector.StrokeLine(screen, x1, y1, x2, y2, t, clr.(interface {
		RGBA() (r, g, b, a uint32)
	}), false)
}

// GetVisibilityState returns the visibility state for rendering purposes.
// This is a convenience wrapper that can be used by the board renderer.
// Returns: 0=shroud, 1=fog, 2=visible
func GetVisibilityState(tile *core.Tile, playerID int) int {
	return tile.VisibilityState(playerID)
}
