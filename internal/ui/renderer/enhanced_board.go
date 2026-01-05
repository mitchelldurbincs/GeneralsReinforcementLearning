package renderer

import (
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"golang.org/x/image/font"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

var (
	SelectionColor   = color.RGBA{255, 255, 100, 255} // Yellow highlight
	ValidMoveColor   = color.RGBA{100, 255, 100, 128} // Semi-transparent green
	HoverColor       = color.RGBA{255, 255, 255, 64}  // Semi-transparent white
	InvalidMoveColor = color.RGBA{255, 100, 100, 64}  // Semi-transparent red
	QueuedMoveColor  = color.RGBA{255, 200, 100, 200} // Orange for queued moves
)

// QueuedMoveInfo represents a queued move for rendering
type QueuedMoveInfo struct {
	FromX, FromY int
	ToX, ToY     int
}

type EnhancedBoardRenderer struct {
	*BoardRenderer

	// Fog renderer for edge highlighting
	fogRenderer *FogRenderer

	// Selection state
	selectedX, selectedY int
	hasSelection         bool

	// Hover state
	hoverX, hoverY int

	// Valid moves cache
	validMoves map[struct{ X, Y int }]bool

	// Queued moves for visualization
	queuedMoves []QueuedMoveInfo

	// Current board and player for visibility checks
	board    *core.Board
	playerID int
}

func NewEnhancedBoardRenderer(tileSize int, f font.Face) *EnhancedBoardRenderer {
	return &EnhancedBoardRenderer{
		BoardRenderer: NewBoardRenderer(tileSize, f),
		fogRenderer:   NewFogRenderer(tileSize),
		validMoves:    make(map[struct{ X, Y int }]bool),
	}
}

func (ebr *EnhancedBoardRenderer) SetSelection(x, y int, hasSelection bool) {
	ebr.selectedX = x
	ebr.selectedY = y
	ebr.hasSelection = hasSelection

	// Update valid moves when selection changes
	if hasSelection {
		ebr.updateValidMoves()
	} else {
		ebr.validMoves = make(map[struct{ X, Y int }]bool)
	}
}

func (ebr *EnhancedBoardRenderer) SetHover(x, y int) {
	ebr.hoverX = x
	ebr.hoverY = y
}

// SetQueuedMoves sets the list of queued moves to render
func (ebr *EnhancedBoardRenderer) SetQueuedMoves(moves []QueuedMoveInfo) {
	ebr.queuedMoves = moves
}

func (ebr *EnhancedBoardRenderer) updateValidMoves() {
	ebr.validMoves = make(map[struct{ X, Y int }]bool)

	// Add orthogonally adjacent tiles as valid moves
	directions := []struct{ dx, dy int }{
		{0, -1}, // up
		{1, 0},  // right
		{0, 1},  // down
		{-1, 0}, // left
	}

	for _, d := range directions {
		x := ebr.selectedX + d.dx
		y := ebr.selectedY + d.dy
		// For now, just mark all adjacent tiles as valid moves
		// The actual validation happens when the move is attempted
		ebr.validMoves[struct{ X, Y int }{x, y}] = true
	}
}

func (ebr *EnhancedBoardRenderer) Draw(screen *ebiten.Image, board *core.Board, players []game.Player, playerID int) {
	// First draw the base board
	ebr.BoardRenderer.Draw(screen, board, players, playerID)

	// Store the board for visibility checks
	ebr.board = board
	ebr.playerID = playerID

	// Then draw overlays
	ebr.drawOverlays(screen, board, playerID)
}

func (ebr *EnhancedBoardRenderer) drawOverlays(screen *ebiten.Image, board *core.Board, playerID int) {
	// Draw visibility edge highlighting first (so it appears under other overlays)
	ebr.fogRenderer.DrawEdgeHighlights(screen, board, playerID)

	// Draw queued moves (so other overlays appear on top)
	for i, move := range ebr.queuedMoves {
		ebr.drawQueuedMove(screen, move, i+1)
	}

	// Draw valid move indicators
	if ebr.hasSelection {
		for move := range ebr.validMoves {
			if board.InBounds(move.X, move.Y) {
				ebr.drawTileOverlay(screen, move.X, move.Y, ValidMoveColor)
			}
		}
	}

	// Draw hover highlight
	if board.InBounds(ebr.hoverX, ebr.hoverY) {
		idx := board.Idx(ebr.hoverX, ebr.hoverY)
		tile := board.T[idx]

		// Show different hover colors based on context
		if ebr.hasSelection {
			if _, isValid := ebr.validMoves[struct{ X, Y int }{ebr.hoverX, ebr.hoverY}]; isValid {
				if !tile.IsMountain() {
					ebr.drawTileOverlay(screen, ebr.hoverX, ebr.hoverY, HoverColor)
				} else {
					ebr.drawTileOverlay(screen, ebr.hoverX, ebr.hoverY, InvalidMoveColor)
				}
			}
		} else {
			// When no selection, only highlight player's own tiles with armies
			if tile.Owner == playerID && tile.Army > 0 && !tile.IsMountain() {
				ebr.drawTileOverlay(screen, ebr.hoverX, ebr.hoverY, HoverColor)
			}
		}
	}

	// Draw selection highlight
	if ebr.hasSelection {
		ebr.drawSelectionBorder(screen, ebr.selectedX, ebr.selectedY)
	}
}

// drawQueuedMove draws an arrow indicating a queued move
func (ebr *EnhancedBoardRenderer) drawQueuedMove(screen *ebiten.Image, move QueuedMoveInfo, moveNumber int) {
	halfTile := float32(ebr.tileSize) / 2

	// Calculate center positions
	fromCenterX := float32(move.FromX*ebr.tileSize) + halfTile
	fromCenterY := float32(move.FromY*ebr.tileSize) + halfTile
	toCenterX := float32(move.ToX*ebr.tileSize) + halfTile
	toCenterY := float32(move.ToY*ebr.tileSize) + halfTile

	// Draw a thick line from source to destination
	lineWidth := float32(4)

	// Calculate direction
	dx := toCenterX - fromCenterX
	dy := toCenterY - fromCenterY

	// Shorten the line slightly so it doesn't go edge to edge
	shortenFactor := float32(0.3)
	startX := fromCenterX + dx*shortenFactor
	startY := fromCenterY + dy*shortenFactor
	endX := toCenterX - dx*shortenFactor
	endY := toCenterY - dy*shortenFactor

	// Draw the line (as a thin rectangle)
	vector.StrokeLine(screen, startX, startY, endX, endY, lineWidth, QueuedMoveColor, false)

	// Draw arrowhead at the end
	arrowSize := float32(8)
	// Normalize direction
	length := float32(ebr.tileSize) * (1 - 2*shortenFactor)
	if length > 0 {
		ndx := dx / length
		ndy := dy / length

		// Arrowhead points
		tipX := endX
		tipY := endY
		leftX := tipX - arrowSize*ndx - arrowSize*ndy*0.5
		leftY := tipY - arrowSize*ndy + arrowSize*ndx*0.5
		rightX := tipX - arrowSize*ndx + arrowSize*ndy*0.5
		rightY := tipY - arrowSize*ndy - arrowSize*ndx*0.5

		// Draw arrowhead as three lines
		vector.StrokeLine(screen, tipX, tipY, leftX, leftY, lineWidth, QueuedMoveColor, false)
		vector.StrokeLine(screen, tipX, tipY, rightX, rightY, lineWidth, QueuedMoveColor, false)
	}
}

func (ebr *EnhancedBoardRenderer) drawTileOverlay(screen *ebiten.Image, gridX, gridY int, c color.Color) {
	screenX := float32(gridX * ebr.tileSize)
	screenY := float32(gridY * ebr.tileSize)
	size := float32(ebr.tileSize)

	vector.DrawFilledRect(screen, screenX, screenY, size, size, c, false)
}

func (ebr *EnhancedBoardRenderer) drawSelectionBorder(screen *ebiten.Image, gridX, gridY int) {
	screenX := float32(gridX * ebr.tileSize)
	screenY := float32(gridY * ebr.tileSize)
	size := float32(ebr.tileSize)
	thickness := float32(3)

	// Draw four border lines
	// Top
	vector.DrawFilledRect(screen, screenX, screenY, size, thickness, SelectionColor, false)
	// Bottom
	vector.DrawFilledRect(screen, screenX, screenY+size-thickness, size, thickness, SelectionColor, false)
	// Left
	vector.DrawFilledRect(screen, screenX, screenY, thickness, size, SelectionColor, false)
	// Right
	vector.DrawFilledRect(screen, screenX+size-thickness, screenY, thickness, size, SelectionColor, false)
}
