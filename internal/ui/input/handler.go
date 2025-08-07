package input

import (
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/rs/zerolog/log"
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
	selectionState    SelectionState
	selectedTileX     int
	selectedTileY     int
	keyboardSelection bool // Track if selection was made via keyboard

	// Movement state
	moveMode     MoveMode
	pendingMoves []PendingMove

	// UI state
	tileSize     int
	boardOffsetX int
	boardOffsetY int

	// Turn state
	isPlayerTurn bool
	turnEnded    bool

	// Validation callback
	tileValidator         func(x, y int) (bool, string)
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
		log.Debug().Msg("Click ignored - not player turn")
		return
	}

	tileX, tileY := h.screenToTile(h.mouseX, h.mouseY)
	log.Debug().
		Int("tileX", tileX).Int("tileY", tileY).
		Int("mouseX", h.mouseX).Int("mouseY", h.mouseY).
		Int("selectionState", int(h.selectionState)).
		Msg("Left click detected")

	switch h.selectionState {
	case SelectionNone:
		// Select a tile if it belongs to the player
		if h.tileValidator != nil {
			if valid, msg := h.tileValidator(tileX, tileY); valid {
				h.selectedTileX = tileX
				h.selectedTileY = tileY
				h.selectionState = SelectionTileSelected
				h.keyboardSelection = false // This is a mouse selection
				h.lastValidationMessage = ""
			} else {
				h.lastValidationMessage = msg
			}
		} else {
			// Fallback to old behavior if no validator
			h.selectedTileX = tileX
			h.selectedTileY = tileY
			h.selectionState = SelectionTileSelected
			h.keyboardSelection = false
		}

	case SelectionTileSelected:
		// If clicking the same tile, deselect
		if tileX == h.selectedTileX && tileY == h.selectedTileY {
			log.Debug().Msg("Deselecting tile - clicked same tile")
			h.selectionState = SelectionNone
		} else {
			// Otherwise, try to move
			move := PendingMove{
				FromX:    h.selectedTileX,
				FromY:    h.selectedTileY,
				ToX:      tileX,
				ToY:      tileY,
				MoveHalf: h.moveMode == MoveHalf,
			}
			h.pendingMoves = append(h.pendingMoves, move)
			log.Debug().
				Int("fromX", move.FromX).Int("fromY", move.FromY).
				Int("toX", move.ToX).Int("toY", move.ToY).
				Bool("moveHalf", move.MoveHalf).
				Int("pendingCount", len(h.pendingMoves)).
				Msg("Created pending move")
			h.selectionState = SelectionNone
		}
	}
}

func (h *Handler) handleRightClick() {
	// Right click cancels selection
	h.selectionState = SelectionNone
}

func (h *Handler) handleKeyboard() {
	// Q for full army movement mode
	if inpututil.IsKeyJustPressed(ebiten.KeyQ) {
		h.moveMode = MoveFull
	}

	// Escape to deselect
	if inpututil.IsKeyJustPressed(ebiten.KeyEscape) {
		h.selectionState = SelectionNone
	}

	// Check for WASD movement if a tile is selected
	if h.selectionState == SelectionTileSelected {
		h.handleMovementKeys()
	}

	// Shift modifier for half army movement (works with WASD too)
	if ebiten.IsKeyPressed(ebiten.KeyShift) {
		h.moveMode = MoveHalf
	} else {
		h.moveMode = MoveFull
	}
}

func (h *Handler) handleMovementKeys() {
	// Direction mapping for WASD and arrow keys
	var dx, dy int
	moved := false

	// WASD keys
	if inpututil.IsKeyJustPressed(ebiten.KeyW) || inpututil.IsKeyJustPressed(ebiten.KeyUp) {
		dx, dy = 0, -1 // North
		moved = true
	} else if inpututil.IsKeyJustPressed(ebiten.KeyA) || inpututil.IsKeyJustPressed(ebiten.KeyLeft) {
		dx, dy = -1, 0 // West
		moved = true
	} else if inpututil.IsKeyJustPressed(ebiten.KeyS) || inpututil.IsKeyJustPressed(ebiten.KeyDown) {
		dx, dy = 0, 1 // South
		moved = true
	} else if inpututil.IsKeyJustPressed(ebiten.KeyD) || inpututil.IsKeyJustPressed(ebiten.KeyRight) {
		dx, dy = 1, 0 // East
		moved = true
	}

	if moved {
		// Create a pending move from selected tile to destination
		toX := h.selectedTileX + dx
		toY := h.selectedTileY + dy

		h.pendingMoves = append(h.pendingMoves, PendingMove{
			FromX:    h.selectedTileX,
			FromY:    h.selectedTileY,
			ToX:      toX,
			ToY:      toY,
			MoveHalf: h.moveMode == MoveHalf,
		})

		log.Debug().
			Str("key", "WASD").
			Int("fromX", h.selectedTileX).Int("fromY", h.selectedTileY).
			Int("toX", toX).Int("toY", toY).
			Bool("moveHalf", h.moveMode == MoveHalf).
			Msg("Created keyboard move")

		// Move selection to the destination tile (actual Generals.io behavior)
		// This allows continuous movement by pressing the same key repeatedly
		h.selectedTileX = toX
		h.selectedTileY = toY
		h.keyboardSelection = true // Mark this as a keyboard-initiated selection
		// Keep selection state active
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
		// Don't clear pending moves here - they should be cleared after processing
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

func (h *Handler) IsKeyboardSelection() bool {
	return h.keyboardSelection
}
