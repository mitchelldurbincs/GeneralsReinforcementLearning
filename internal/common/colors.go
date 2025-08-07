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
