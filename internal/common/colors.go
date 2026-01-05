package common

import (
	"image/color"
)

// PlayerColors defines the color scheme for each player
var PlayerColors = map[int]color.Color{
	-1: color.RGBA{120, 120, 120, 255}, // Neutral – gray
	0:  color.RGBA{200, 50, 50, 255},   // Red
	1:  color.RGBA{50, 100, 200, 255},  // Blue
	2:  color.RGBA{50, 200, 50, 255},   // Green
	3:  color.RGBA{200, 200, 50, 255},  // Yellow
}

// PlayerColorsLight - brighter variants for highlights and hover effects
var PlayerColorsLight = map[int]color.Color{
	-1: color.RGBA{160, 160, 160, 255}, // Light gray
	0:  color.RGBA{255, 100, 100, 255}, // Light red
	1:  color.RGBA{100, 150, 255, 255}, // Light blue
	2:  color.RGBA{100, 255, 100, 255}, // Light green
	3:  color.RGBA{255, 255, 100, 255}, // Light yellow
}

// PlayerColorsDark - darker variants for shadows and depth
var PlayerColorsDark = map[int]color.Color{
	-1: color.RGBA{80, 80, 80, 255},   // Dark gray
	0:  color.RGBA{140, 30, 30, 255},  // Dark red
	1:  color.RGBA{30, 60, 140, 255},  // Dark blue
	2:  color.RGBA{30, 140, 30, 255},  // Dark green
	3:  color.RGBA{140, 140, 30, 255}, // Dark yellow
}

// Tile colors
var (
	MountainColor        = color.RGBA{80, 80, 80, 255}
	CityOwnedHueShift    = 30
	GeneralSymbolColor   = color.White
	CitySymbolColor      = color.White
	ArmyTextColor        = color.White
	GeneralArmyTextColor = color.Black
)

// UI colors
var (
	BackgroundColor = color.Black
	GridLineColor   = color.RGBA{50, 50, 50, 255}
	FogOfWarColor   = color.RGBA{0, 0, 0, 200}
)

// Fog of War visibility colors
var (
	// ShroudColor is for tiles never seen (darkest)
	ShroudColor = color.RGBA{10, 10, 10, 230}
	// FogColor is for tiles previously seen but not currently visible (lighter than shroud)
	FogColor = color.RGBA{40, 40, 40, 180}
	// VisibilityEdgeColor is the highlight color at visibility boundaries
	VisibilityEdgeColor = color.RGBA{255, 255, 100, 180} // Yellow-ish glow
)
