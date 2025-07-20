package game

import (
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/config"
)

// Map generation functions
func CityRatio() int {
	return config.Get().Game.Map.CityRatio
}

func CityStartArmy() int {
	return config.Get().Game.Map.CityStartArmy
}

func MinGeneralSpacing() int {
	return config.Get().Game.Map.MinGeneralSpacing
}

// Production rate functions
func GeneralProduction() int {
	return config.Get().Game.Production.General
}

func CityProduction() int {
	return config.Get().Game.Production.City
}

func NormalProduction() int {
	return config.Get().Game.Production.Normal
}

func NormalGrowInterval() int {
	return config.Get().Game.Production.NormalGrowthInterval
}