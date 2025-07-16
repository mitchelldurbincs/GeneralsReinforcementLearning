package common

import (
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPlayerColors(t *testing.T) {
	tests := []struct {
		playerID     int
		expectedName string
		checkColor   func(color.Color) bool
	}{
		{
			playerID:     -1,
			expectedName: "neutral gray",
			checkColor: func(c color.Color) bool {
				rgba := c.(color.RGBA)
				// Gray color should have equal RGB components
				return rgba.R == rgba.G && rgba.G == rgba.B && rgba.R == 120
			},
		},
		{
			playerID:     0,
			expectedName: "player 0 red",
			checkColor: func(c color.Color) bool {
				rgba := c.(color.RGBA)
				// Red should have high R, low G and B
				return rgba.R > rgba.G && rgba.R > rgba.B
			},
		},
		{
			playerID:     1,
			expectedName: "player 1 blue",
			checkColor: func(c color.Color) bool {
				rgba := c.(color.RGBA)
				// Blue should have high B, lower R and G
				return rgba.B > rgba.R && rgba.B > rgba.G
			},
		},
		{
			playerID:     2,
			expectedName: "player 2 green",
			checkColor: func(c color.Color) bool {
				rgba := c.(color.RGBA)
				// Green should have high G, lower R and B
				return rgba.G > rgba.R && rgba.G > rgba.B
			},
		},
		{
			playerID:     3,
			expectedName: "player 3 yellow",
			checkColor: func(c color.Color) bool {
				rgba := c.(color.RGBA)
				// Yellow should have high R and G, low B
				return rgba.R > 150 && rgba.G > 150 && rgba.B < 100
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.expectedName, func(t *testing.T) {
			c, exists := PlayerColors[tt.playerID]
			assert.True(t, exists, "color should exist for player %d", tt.playerID)
			assert.NotNil(t, c)
			assert.True(t, tt.checkColor(c), "color check failed for player %d", tt.playerID)
			
			// All colors should be fully opaque
			rgba, ok := c.(color.RGBA)
			assert.True(t, ok, "color should be RGBA type")
			assert.Equal(t, uint8(255), rgba.A, "alpha should be 255 (fully opaque)")
		})
	}
}

func TestPlayerColorsCompleteness(t *testing.T) {
	// Ensure we have colors for players -1 through 3
	for i := -1; i <= 3; i++ {
		_, exists := PlayerColors[i]
		assert.True(t, exists, "color should exist for player %d", i)
	}
	
	// Check that we don't have extra colors
	assert.Len(t, PlayerColors, 5, "should have exactly 5 player colors (-1 through 3)")
}

func TestTileColors(t *testing.T) {
	t.Run("mountain color", func(t *testing.T) {
		assert.NotNil(t, MountainColor)
		// Mountain should be dark gray
		assert.Equal(t, uint8(80), MountainColor.R)
		assert.Equal(t, uint8(80), MountainColor.G)
		assert.Equal(t, uint8(80), MountainColor.B)
		assert.Equal(t, uint8(255), MountainColor.A)
	})
	
	t.Run("city hue shift", func(t *testing.T) {
		assert.Equal(t, 30, CityOwnedHueShift)
	})
	
	t.Run("symbol colors", func(t *testing.T) {
		// General and city symbols should be white
		assert.Equal(t, color.White, GeneralSymbolColor)
		assert.Equal(t, color.White, CitySymbolColor)
		assert.Equal(t, color.White, ArmyTextColor)
		
		// General army text should be black for contrast
		assert.Equal(t, color.Black, GeneralArmyTextColor)
	})
}

func TestUIColors(t *testing.T) {
	t.Run("background color", func(t *testing.T) {
		assert.Equal(t, color.Black, BackgroundColor)
	})
	
	t.Run("grid line color", func(t *testing.T) {
		assert.NotNil(t, GridLineColor)
		// Grid lines should be dark gray
		assert.Equal(t, uint8(50), GridLineColor.R)
		assert.Equal(t, uint8(50), GridLineColor.G)
		assert.Equal(t, uint8(50), GridLineColor.B)
		assert.Equal(t, uint8(255), GridLineColor.A)
	})
	
	t.Run("fog of war color", func(t *testing.T) {
		assert.NotNil(t, FogOfWarColor)
		// Fog should be semi-transparent black
		assert.Equal(t, uint8(0), FogOfWarColor.R)
		assert.Equal(t, uint8(0), FogOfWarColor.G)
		assert.Equal(t, uint8(0), FogOfWarColor.B)
		assert.Equal(t, uint8(200), FogOfWarColor.A) // Semi-transparent
	})
}

func TestColorConsistency(t *testing.T) {
	// Ensure no two player colors are identical
	t.Run("unique player colors", func(t *testing.T) {
		colorMap := make(map[color.RGBA]int)
		for playerID, c := range PlayerColors {
			rgba, ok := c.(color.RGBA)
			assert.True(t, ok, "color should be RGBA type")
			if existingID, exists := colorMap[rgba]; exists {
				t.Errorf("players %d and %d have the same color", existingID, playerID)
			}
			colorMap[rgba] = playerID
		}
	})
	
	// Ensure colors provide good contrast
	t.Run("color contrast", func(t *testing.T) {
		// Check that player colors are distinguishable from background
		for playerID, c := range PlayerColors {
			rgba, ok := c.(color.RGBA)
			assert.True(t, ok, "color should be RGBA type")
			// All player colors should have at least one component > 50
			// to be visible against black background
			maxComponent := rgba.R
			if rgba.G > maxComponent {
				maxComponent = rgba.G
			}
			if rgba.B > maxComponent {
				maxComponent = rgba.B
			}
			assert.Greater(t, maxComponent, uint8(50), 
				"player %d color should be visible against black background", playerID)
		}
	})
}

func TestColorAccessibility(t *testing.T) {
	// Test that getting a non-existent player color returns zero value
	t.Run("non-existent player color", func(t *testing.T) {
		_, exists := PlayerColors[99]
		assert.False(t, exists, "color should not exist for player 99")
	})
}