package game

import (
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/rs/zerolog"
)

// ProductionManager handles army production for all tile types
type ProductionManager struct {
	eventBus *events.EventBus
	gameID   string
	logger   zerolog.Logger
}

// NewProductionManager creates a new production manager
func NewProductionManager(eventBus *events.EventBus, gameID string, logger zerolog.Logger) *ProductionManager {
	return &ProductionManager{
		eventBus: eventBus,
		gameID:   gameID,
		logger:   logger.With().Str("component", "ProductionManager").Logger(),
	}
}

// ProcessTurnProduction applies army growth for the given turn
func (pm *ProductionManager) ProcessTurnProduction(gs *GameState, turn int) {
	growNormal := turn%NormalGrowInterval() == 0
	pm.logger.Debug().
		Int("turn", turn).
		Bool("grow_normal_tiles", growNormal).
		Msg("Processing turn production")

	// Track production totals
	totalGeneralProd := 0
	totalCityProd := 0
	totalNormalProd := 0

	// Iterate only through tiles owned by players
	for pid := range gs.Players {
		if !gs.Players[pid].Alive {
			continue
		}

		// Process production for each owned tile
		for _, tileIdx := range gs.Players[pid].OwnedTiles {
			prod := pm.processTileProduction(&gs.Board.T[tileIdx], growNormal)

			// Track production by type
			switch gs.Board.T[tileIdx].Type {
			case core.TileGeneral:
				totalGeneralProd += prod
			case core.TileCity:
				totalCityProd += prod
			case core.TileNormal:
				totalNormalProd += prod
			}

			// Mark tile as changed if production occurred
			if prod > 0 {
				gs.ChangedTiles[tileIdx] = struct{}{}
			}
		}
	}

	// Publish production event if any production occurred
	pm.publishProductionEvent(totalNormalProd, totalCityProd, totalGeneralProd, turn)

	pm.logger.Debug().
		Int("total_general_production", totalGeneralProd).
		Int("total_city_production", totalCityProd).
		Int("total_normal_production", totalNormalProd).
		Msg("Turn production complete")
}

// processTileProduction applies production to a single tile
func (pm *ProductionManager) processTileProduction(tile *core.Tile, growNormal bool) int {
	switch tile.Type {
	case core.TileGeneral:
		production := GeneralProduction()
		tile.Army += production
		return production

	case core.TileCity:
		production := CityProduction()
		tile.Army += production
		return production

	case core.TileNormal:
		if growNormal {
			production := NormalProduction()
			tile.Army += production
			return production
		}

	// Mountains don't produce
	case core.TileMountain:
		return 0
	}

	return 0
}

// publishProductionEvent publishes a production event if any production occurred
func (pm *ProductionManager) publishProductionEvent(normalProd, cityProd, generalProd, turn int) {
	if normalProd > 0 || cityProd > 0 || generalProd > 0 {
		pm.eventBus.Publish(events.NewProductionAppliedEvent(
			pm.gameID,
			normalProd,
			cityProd,
			generalProd,
			turn,
		))
	}
}
