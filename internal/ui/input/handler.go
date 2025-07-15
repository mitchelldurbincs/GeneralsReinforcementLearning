package input

import (
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

type SelectionState int

const (
	SelectionNone SelectionState = iota
	SelectionTileSelected
	SelectionMovePending
)

type MoveMode int

const (
	MoveFull MoveMode = iota
	MoveHalf
)

type Handler struct {
	// Mouse state
	mouseX, mouseY int
	
	// Selection state
	selectionState SelectionState
	selectedTileX  int
	selectedTileY  int
	
	// Movement state
	moveMode       MoveMode
	pendingMoves   []PendingMove
	
	// UI state
	tileSize       int
	boardOffsetX   int
	boardOffsetY   int
	
	// Turn state
	isPlayerTurn   bool
	turnEnded      bool
	
	// Validation callback
	tileValidator  func(x, y int) (bool, string)
	lastValidationMessage string
}

type PendingMove struct {
	FromX, FromY int
	ToX, ToY     int
	MoveHalf     bool
}

func NewHandler(tileSize int) *Handler {
	return &Handler{
		tileSize:       tileSize,
		selectionState: SelectionNone,
		moveMode:       MoveFull,
		pendingMoves:   make([]PendingMove, 0),
	}
}

func (h *Handler) Update() {
	// Update mouse position
	h.mouseX, h.mouseY = ebiten.CursorPosition()
	
	// Handle mouse clicks
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonLeft) {
		h.handleLeftClick()
	}
	
	if inpututil.IsMouseButtonJustPressed(ebiten.MouseButtonRight) {
		h.handleRightClick()
	}
	
	// Handle keyboard input
	h.handleKeyboard()
}

func (h *Handler) handleLeftClick() {
	if !h.isPlayerTurn {
		return
	}
	
	tileX, tileY := h.screenToTile(h.mouseX, h.mouseY)
	
	switch h.selectionState {
	case SelectionNone:
		// Select a tile if it belongs to the player
		if h.tileValidator != nil {
			if valid, msg := h.tileValidator(tileX, tileY); valid {
				h.selectedTileX = tileX
				h.selectedTileY = tileY
				h.selectionState = SelectionTileSelected
				h.lastValidationMessage = ""
			} else {
				h.lastValidationMessage = msg
			}
		} else {
			// Fallback to old behavior if no validator
			h.selectedTileX = tileX
			h.selectedTileY = tileY
			h.selectionState = SelectionTileSelected
		}
		
	case SelectionTileSelected:
		// If clicking the same tile, deselect
		if tileX == h.selectedTileX && tileY == h.selectedTileY {
			h.selectionState = SelectionNone
		} else {
			// Otherwise, try to move
			h.pendingMoves = append(h.pendingMoves, PendingMove{
				FromX:    h.selectedTileX,
				FromY:    h.selectedTileY,
				ToX:      tileX,
				ToY:      tileY,
				MoveHalf: h.moveMode == MoveHalf,
			})
			h.selectionState = SelectionNone
		}
	}
}

func (h *Handler) handleRightClick() {
	// Right click cancels selection
	h.selectionState = SelectionNone
}

func (h *Handler) handleKeyboard() {
	// Q for full army movement
	if inpututil.IsKeyJustPressed(ebiten.KeyQ) {
		h.moveMode = MoveFull
	}
	
	// W for half army movement
	if inpututil.IsKeyJustPressed(ebiten.KeyW) {
		h.moveMode = MoveHalf
	}
	
	// Escape to deselect
	if inpututil.IsKeyJustPressed(ebiten.KeyEscape) {
		h.selectionState = SelectionNone
	}
	
	// Shift modifier for half army movement
	if ebiten.IsKeyPressed(ebiten.KeyShift) {
		h.moveMode = MoveHalf
	} else if !inpututil.IsKeyJustPressed(ebiten.KeyW) {
		h.moveMode = MoveFull
	}
}

func (h *Handler) screenToTile(x, y int) (int, int) {
	tileX := (x - h.boardOffsetX) / h.tileSize
	tileY := (y - h.boardOffsetY) / h.tileSize
	return tileX, tileY
}

func (h *Handler) SetBoardOffset(x, y int) {
	h.boardOffsetX = x
	h.boardOffsetY = y
}

func (h *Handler) SetPlayerTurn(isTurn bool) {
	h.isPlayerTurn = isTurn
	if isTurn {
		h.turnEnded = false
		h.pendingMoves = h.pendingMoves[:0] // Clear pending moves
	}
}

func (h *Handler) GetSelectedTile() (int, int, bool) {
	if h.selectionState == SelectionTileSelected {
		return h.selectedTileX, h.selectedTileY, true
	}
	return 0, 0, false
}

func (h *Handler) GetHoveredTile() (int, int) {
	return h.screenToTile(h.mouseX, h.mouseY)
}

func (h *Handler) GetPendingMoves() []PendingMove {
	return h.pendingMoves
}

func (h *Handler) ClearPendingMoves() {
	h.pendingMoves = h.pendingMoves[:0]
}

func (h *Handler) IsTurnEnded() bool {
	return h.turnEnded
}

func (h *Handler) GetMoveMode() MoveMode {
	return h.moveMode
}

func (h *Handler) SetTileValidator(validator func(x, y int) (bool, string)) {
	h.tileValidator = validator
}

func (h *Handler) GetLastValidationMessage() string {
	msg := h.lastValidationMessage
	h.lastValidationMessage = "" // Clear after reading
	return msg
}