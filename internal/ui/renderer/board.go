package renderer

import (
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"golang.org/x/image/font"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/common"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

// -----------------------------------------------------------------------------
// Tile Cache - Pre-rendered tile images to avoid per-frame allocations
// -----------------------------------------------------------------------------

// Visibility states for tile rendering
const (
	VisShroud  = 0 // Never seen (darkest)
	VisFog     = 1 // Previously seen but not currently visible
	VisVisible = 2 // Currently visible
)

const (
	numOwners    = 5 // -1 (neutral) + 4 players (0-3)
	numTileTypes = 4 // TileNormal, TileGeneral, TileCity, TileMountain
	numVisStates = 3 // shroud, fog, visible
)

// TileCache holds pre-rendered tile images for all possible tile states.
// This eliminates the need to create new images every frame.
type TileCache struct {
	tileSize int

	// baseTiles[ownerIdx][tileType][visState] where:
	// - ownerIdx = owner + 1 (so -1 becomes 0, 0 becomes 1, etc.)
	// - tileType = TileNormal(0), TileGeneral(1), TileCity(2), TileMountain(3)
	// - visState = 0 (shroud), 1 (fog), 2 (visible)
	baseTiles [numOwners][numTileTypes][numVisStates]*ebiten.Image

	// Overlay images for different visibility states
	shroudOverlay *ebiten.Image // Very dark, never seen
	fogOverlay    *ebiten.Image // Lighter, previously seen

	// Texture overlays for visual depth
	normalTexture   *ebiten.Image // Subtle grid pattern for normal tiles
	mountainTexture *ebiten.Image // Diagonal lines for mountains
}

// NewTileCache creates a new tile cache with pre-rendered images for all tile states.
func NewTileCache(tileSize int) *TileCache {
	tc := &TileCache{tileSize: tileSize}
	tc.init()
	return tc
}

func (tc *TileCache) init() {
	// Pre-render shroud overlay (very dark, never seen)
	tc.shroudOverlay = ebiten.NewImage(tc.tileSize, tc.tileSize)
	tc.shroudOverlay.Fill(common.ShroudColor)

	// Pre-render fog overlay (lighter, previously seen)
	tc.fogOverlay = ebiten.NewImage(tc.tileSize, tc.tileSize)
	tc.fogOverlay.Fill(common.FogColor)

	// Pre-render texture overlays
	tc.normalTexture = tc.generateNormalTexture()
	tc.mountainTexture = tc.generateMountainTexture()

	// Pre-render all tile combinations for all 3 visibility states
	for owner := -1; owner <= 3; owner++ {
		for tileType := 0; tileType < numTileTypes; tileType++ {
			for visState := 0; visState < numVisStates; visState++ {
				tc.baseTiles[owner+1][tileType][visState] = tc.renderTile(owner, tileType, visState)
			}
		}
	}
}

// generateNormalTexture creates a subtle grid pattern for normal tiles
func (tc *TileCache) generateNormalTexture() *ebiten.Image {
	img := ebiten.NewImage(tc.tileSize, tc.tileSize)
	lineColor := color.RGBA{0, 0, 0, 15} // Very subtle dark lines
	size := float32(tc.tileSize)

	// Draw subtle grid lines
	for i := 4; i < tc.tileSize; i += 4 {
		fi := float32(i)
		vector.StrokeLine(img, fi, 0, fi, size, 1, lineColor, false)
		vector.StrokeLine(img, 0, fi, size, fi, 1, lineColor, false)
	}
	return img
}

// generateMountainTexture creates diagonal lines for mountain tiles
func (tc *TileCache) generateMountainTexture() *ebiten.Image {
	img := ebiten.NewImage(tc.tileSize, tc.tileSize)
	lineColor := color.RGBA{40, 40, 40, 60} // Subtle diagonal lines
	size := float32(tc.tileSize)

	// Draw diagonal lines from top-left to bottom-right
	for i := -tc.tileSize; i < tc.tileSize*2; i += 6 {
		fi := float32(i)
		vector.StrokeLine(img, fi, 0, fi+size, size, 1, lineColor, false)
	}
	return img
}

// renderTile creates a single pre-rendered tile image.
// visState: 0=shroud (never seen), 1=fog (previously seen), 2=visible
func (tc *TileCache) renderTile(owner, tileType, visState int) *ebiten.Image {
	cell := ebiten.NewImage(tc.tileSize, tc.tileSize)
	visible := visState == VisVisible

	// Mountain tiles
	if tileType == core.TileMountain {
		cell.Fill(common.MountainColor)
		// Apply mountain texture for visual depth
		cell.DrawImage(tc.mountainTexture, nil)
		// Apply appropriate overlay based on visibility state
		switch visState {
		case VisShroud:
			cell.DrawImage(tc.shroudOverlay, nil)
		case VisFog:
			cell.DrawImage(tc.fogOverlay, nil)
		}
		return cell
	}

	// Determine base color - non-visible tiles show neutral color
	tileColor := common.PlayerColors[core.NeutralID]
	if visible {
		if c, ok := common.PlayerColors[owner]; ok {
			tileColor = c
		}
	}
	cell.Fill(tileColor)

	// Apply subtle grid texture for normal tiles
	if tileType == core.TileNormal {
		cell.DrawImage(tc.normalTexture, nil)
	}

	// Owned city inner square (darker shade) - only when visible
	if visible && tileType == core.TileCity && owner != core.NeutralID {
		m := tc.tileSize / 3
		sq := ebiten.NewImage(m, m)
		sq.Fill(shiftColor(tileColor, common.CityOwnedHueShift))
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(float64(tc.tileSize-m)/2, float64(tc.tileSize-m)/2)
		cell.DrawImage(sq, op)
	}

	// City marker (gray square) - only when visible
	if visible && tileType == core.TileCity {
		m := tc.tileSize / 2
		sq := ebiten.NewImage(m, m)
		sq.Fill(color.RGBA{180, 180, 180, 255})
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(float64(tc.tileSize-m)/2, float64(tc.tileSize-m)/2)
		cell.DrawImage(sq, op)
	}

	// General marker (white square) - only when visible
	if visible && tileType == core.TileGeneral {
		m := tc.tileSize / 2
		sq := ebiten.NewImage(m, m)
		sq.Fill(common.GeneralSymbolColor)
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(float64(tc.tileSize-m)/2, float64(tc.tileSize-m)/2)
		cell.DrawImage(sq, op)
	}

	// Apply appropriate overlay based on visibility state
	switch visState {
	case VisShroud:
		cell.DrawImage(tc.shroudOverlay, nil)
	case VisFog:
		cell.DrawImage(tc.fogOverlay, nil)
	}

	return cell
}

// GetTile returns the pre-rendered image for a tile with the given properties.
// visState: 0=shroud, 1=fog, 2=visible (use VisShroud, VisFog, VisVisible constants)
func (tc *TileCache) GetTile(owner, tileType, visState int) *ebiten.Image {
	ownerIdx := owner + 1
	if ownerIdx < 0 || ownerIdx >= numOwners {
		ownerIdx = 0 // Default to neutral
	}
	if tileType < 0 || tileType >= numTileTypes {
		tileType = core.TileNormal
	}
	if visState < 0 || visState >= numVisStates {
		visState = VisShroud
	}
	return tc.baseTiles[ownerIdx][tileType][visState]
}

// -----------------------------------------------------------------------------
// Board Renderer
// -----------------------------------------------------------------------------

type BoardRenderer struct {
	tileSize     int
	defaultFont  font.Face
	cache        *TileCache
	armyRenderer *ArmyRenderer
}

// NewBoardRenderer returns a renderer ready to use.
// It pre-renders all tile variations into a cache for optimal performance.
func NewBoardRenderer(tileSize int, f font.Face) *BoardRenderer {
	return &BoardRenderer{
		tileSize:     tileSize,
		defaultFont:  f,
		cache:        NewTileCache(tileSize),
		armyRenderer: NewArmyRenderer(f),
	}
}

// Draw renders the board on the supplied Ebiten screen.
// Uses pre-rendered tile cache for optimal performance (no per-frame allocations).
// Supports 3 visibility states: shroud (never seen), fog (previously seen), visible.
func (br *BoardRenderer) Draw(screen *ebiten.Image, board *core.Board, players []game.Player, playerID int) {
	if board == nil {
		return
	}

	for i, tile := range board.T {
		gridX, gridY := board.XY(i)

		screenX := float64(gridX * br.tileSize)
		screenY := float64(gridY * br.tileSize)

		// Determine visibility state (0=shroud, 1=fog, 2=visible)
		visState := tile.VisibilityState(playerID)

		// Get pre-rendered tile from cache (no allocation)
		cachedTile := br.cache.GetTile(tile.Owner, tile.Type, visState)

		// Blit cached tile to screen
		op := &ebiten.DrawImageOptions{}
		op.GeoM.Translate(screenX, screenY)
		screen.DrawImage(cachedTile, op)

		// Army count text with outline effect (rendered per-frame since values change)
		// Only show for visible tiles
		if visState == VisVisible && tile.Army > 0 && !tile.IsMountain() {
			br.armyRenderer.DrawArmyCount(screen, int(screenX), int(screenY), tile.Army, tile.IsGeneral(), br.tileSize)
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
