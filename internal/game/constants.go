package game

// Game balance constants
const (
	// Map generation
	CityRatio         = 20 // 1 city per 20 tiles (≈5%)
	CityStartArmy     = 40
	MinGeneralSpacing = 5 // Manhattan distance between generals

	// Production rates
	GeneralProduction  = 1  // armies per turn
	CityProduction     = 1  // armies per turn (when owned)
	NormalProduction   = 1  // armies per interval
	NormalGrowInterval = 25 // turns between normal tile growth
)