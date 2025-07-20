package game

import (
	"context"
	"errors" 
	"math/rand"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/mapgen"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/processor"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/rules"
	"github.com/rs/zerolog"
)

type Engine struct {
	gs       *GameState
	rng      *rand.Rand
	gameOver bool
	logger   zerolog.Logger
	
	// Extracted components
	actionProcessor *processor.ActionProcessor
	winCondition    *rules.WinConditionChecker
	legalMoves      *rules.LegalMoveCalculator
	
	// Reusable temporary maps to avoid allocations
	tempTileOwnership   map[int]int        // Used in performIncrementalStatsUpdate
	tempAffectedPlayers map[int]struct{}   // Used in performIncrementalVisibilityUpdate
}

type GameConfig struct {
	Width    int
	Height   int
	Players  int
	Rng      *rand.Rand
	Logger   zerolog.Logger
}

// NewEngine creates a new game engine with map generation.
// Accepts a context, primarily for consistency and potential future use (e.g., complex setup).
func NewEngine(ctx context.Context, cfg GameConfig) *Engine {
	// Create a logger for this engine instance, potentially deriving from a passed-in logger
	// or using a default if none is provided.
	// For this example, we'll use the parentLogger and add engine-specific fields.
	engineLogger := cfg.Logger.With().Str("component", "GameEngine").Logger()

	// Example: Check context if setup were long (not strictly necessary here yet)
	select {
	case <-ctx.Done():
		engineLogger.Error().Err(ctx.Err()).Msg("Engine creation cancelled or timed out during initial phase")
		return nil
	default:
	}

	// Set a default RNG if none provided
	if cfg.Rng == nil {
		engineLogger.Debug().Msg("No RNG provided, creating new seeded RNG")
		cfg.Rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Generate the map
	mapCfg := mapgen.DefaultMapConfig(cfg.Width, cfg.Height, cfg.Players)
	generator := mapgen.NewGenerator(mapCfg, cfg.Rng)
	board := generator.GenerateMap()
	if board == nil {
		engineLogger.Error().Msg("Map generation failed")
		return nil
	}

	// Initialize game state with the generated map
	gs := &GameState{
		Board:                 board,
		Players:               make([]Player, cfg.Players),
		Turn:                  0,
		FogOfWarEnabled:       true,
		ChangedTiles:          make(map[int]struct{}),
		VisibilityChangedTiles: make(map[int]struct{}),
	}

	// Initialize players and find their generals
	for i := 0; i < cfg.Players; i++ {
		generalIdx := -1
		// Find this player's general on the board
		for idx, tile := range board.T {
			if tile.Type == core.TileGeneral && tile.Owner == i {
				generalIdx = idx
				break
			}
		}
		
		gs.Players[i] = Player{
			ID:         i,
			Alive:      true,
			GeneralIdx: generalIdx,
			ArmyCount:  1,
			OwnedTiles: make([]int, 0, 50), // Pre-allocate some capacity
		}
	}

	e := &Engine{
		gs:       gs,
		rng:      cfg.Rng,
		gameOver: false,
		logger:   engineLogger,
		actionProcessor: processor.NewActionProcessor(engineLogger),
		winCondition:    rules.NewWinConditionChecker(engineLogger, cfg.Players),
		legalMoves:      rules.NewLegalMoveCalculator(),
		tempTileOwnership:   make(map[int]int),
		tempAffectedPlayers: make(map[int]struct{}),
	}

	// Initial update of player stats to populate OwnedTiles
	e.updatePlayerStats()
	e.updateFogOfWar()
	// CheckGameOver on initialization is probably not needed unless a 0-player game is valid.
	e.checkGameOver(engineLogger.With().Str("phase", "init").Logger())
	engineLogger.Info().Int("width", cfg.Width).Int("height", cfg.Height).Int("players", cfg.Players).Msg("Engine created successfully")
	return e
}

// Step processes actions and advances the game by one turn.
// It accepts a context for cancellation/timeout.
func (e *Engine) Step(ctx context.Context, actions []core.Action) error {
	// At the beginning of the step, check for cancellation.
	select {
	case <-ctx.Done():
		e.logger.Warn().Err(ctx.Err()).Int("turn", e.gs.Turn).Msg("Game step cancelled or timed out before starting")
		return ctx.Err()
	default:
	}

	if e.gameOver {
		e.logger.Warn().Int("turn", e.gs.Turn).Msg("Attempted to step game that is already over")
		return core.ErrGameOver
	}

	e.gs.Turn++
	e.updateFogOfWar()
	// Clear changed tiles from previous turn (reuse the map to avoid allocation)
	for k := range e.gs.ChangedTiles {
		delete(e.gs.ChangedTiles, k)
	}
	for k := range e.gs.VisibilityChangedTiles {
		delete(e.gs.VisibilityChangedTiles, k)
	}
	
	// You can derive the turnLogger from e.logger or potentially from a logger in ctx.
	// For simplicity, we continue deriving from e.logger.
	// If ctx contained a request-specific logger: `baseLoggerForTurn := zerolog.Ctx(ctx)`
	// then `turnLogger := baseLoggerForTurn.With().Int("turn", e.gs.Turn).Logger()`
	turnLogger := e.logger.With().Int("turn", e.gs.Turn).Logger()

	turnLogger.Debug().Msg("Starting game step")

	turnLogger.Debug().Int("num_actions_submitted", len(actions)).Msg("Processing actions")
	if err := e.processActions(ctx, actions, turnLogger); err != nil {
		// Check if the error is due to context cancellation, which might have already been logged by processActions
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err // Already logged, just propagate
		}
		// Other errors from processActions would have been logged there.
		return err
	}
	turnLogger.Debug().Msg("Finished processing actions")

	// Check context again before potentially long operations
	select {
	case <-ctx.Done():
		turnLogger.Warn().Err(ctx.Err()).Msg("Game step cancelled or timed out before production")
		return ctx.Err()
	default:
	}
	e.processTurnProduction(turnLogger)

	select {
	case <-ctx.Done():
		turnLogger.Warn().Err(ctx.Err()).Msg("Game step cancelled or timed out before updating/checking stats")
		return ctx.Err()
	default:
	}
	e.updatePlayerStats()
	e.checkGameOver(turnLogger)

	turnLogger.Debug().Msg("Game step finished")
	return nil
}

// processActions handles all actions for this turn using the ActionProcessor.
func (e *Engine) processActions(ctx context.Context, actions []core.Action, l zerolog.Logger) error {
	// Create a slice of PlayerInfo interfaces from our Players
	playerInfos := make([]processor.PlayerInfo, len(e.gs.Players))
	for i := range e.gs.Players {
		playerInfos[i] = &e.gs.Players[i]
	}
	
	// Use the ActionProcessor to handle all action processing
	captureDetails, visibilityChangedTiles, err := e.actionProcessor.ProcessActions(ctx, e.gs.Board, playerInfos, actions, e.gs.ChangedTiles)
	
	// Merge visibility changed tiles
	for tileIdx := range visibilityChangedTiles {
		e.gs.VisibilityChangedTiles[tileIdx] = struct{}{}
	}
	
	// Handle eliminations if there were captures
	if len(captureDetails) > 0 {
		l.Debug().Int("num_captures_this_turn", len(captureDetails)).Msg("Processing captures for eliminations")
		eliminationOrders := core.ProcessCaptures(captureDetails)
		if len(eliminationOrders) > 0 {
			e.handleEliminationsAndTileTurnover(eliminationOrders, l)
			// Update player stats immediately after eliminations so production is applied correctly
			e.updatePlayerStats()
		}
	}

	return err
}

// handleEliminationsAndTileTurnover processes player eliminations and transfers tiles.
func (e *Engine) handleEliminationsAndTileTurnover(orders []core.PlayerEliminationOrder, l zerolog.Logger) {
	l.Info().Int("num_elimination_orders", len(orders)).Msg("Handling player eliminations and tile turnover")
	for _, order := range orders {
		l.Info().
			Int("eliminated_player_id", order.EliminatedPlayerID).
			Int("new_owner_player_id", order.NewOwnerID).
			Msg("Player eliminated, transferring assets")

		// Use the owned tiles list for efficient transfer
		eliminatedPlayer := &e.gs.Players[order.EliminatedPlayerID]
		tilesTransferred := 0
		
		for _, tileIdx := range eliminatedPlayer.OwnedTiles {
			if e.gs.Board.T[tileIdx].Owner == order.EliminatedPlayerID {
				e.gs.Board.T[tileIdx].Owner = order.NewOwnerID
				e.gs.ChangedTiles[tileIdx] = struct{}{}
				e.gs.VisibilityChangedTiles[tileIdx] = struct{}{}
				tilesTransferred++
			}
		}
		
		// Mark the player as eliminated
		eliminatedPlayer.Alive = false
		eliminatedPlayer.GeneralIdx = -1
		
		l.Debug().
			Int("eliminated_player_id", order.EliminatedPlayerID).
			Int("new_owner_player_id", order.NewOwnerID).
			Int("tiles_transferred", tilesTransferred).
			Msg("Asset transfer details")
	}
}

// processTurnProduction applies army growth
func (e *Engine) processTurnProduction(l zerolog.Logger) {
	growNormal := e.gs.Turn%NormalGrowInterval == 0
	l.Debug().Bool("grow_normal_tiles", growNormal).Msg("Processing turn production")

	// Could add more detailed logs here if needed, e.g., total production amounts
	// For now, keeping it concise as per-tile logging would be too verbose for INFO/DEBUG
	var totalGeneralProd, totalCityProd, totalNormalProd int

	// Iterate only through tiles owned by players
	for pid := range e.gs.Players {
		if !e.gs.Players[pid].Alive {
			continue
		}
		
		for _, tileIdx := range e.gs.Players[pid].OwnedTiles {
			t := &e.gs.Board.T[tileIdx]
			
			switch t.Type {
			case core.TileGeneral:
				t.Army += GeneralProduction
				totalGeneralProd += GeneralProduction
				e.gs.ChangedTiles[tileIdx] = struct{}{}
			case core.TileCity:
				t.Army += CityProduction
				totalCityProd += CityProduction
				e.gs.ChangedTiles[tileIdx] = struct{}{}
			case core.TileNormal:
				if growNormal {
					t.Army += NormalProduction
					totalNormalProd += NormalProduction
					e.gs.ChangedTiles[tileIdx] = struct{}{}
				}
			}
		}
	}
	l.Debug().
		Int("total_general_production", totalGeneralProd).
		Int("total_city_production", totalCityProd).
		Int("total_normal_production", totalNormalProd).
		Msg("Turn production complete")
}

// checkGameOver determines if the game is over using the WinConditionChecker.
func (e *Engine) checkGameOver(l zerolog.Logger) {
	// Create a slice of Player interfaces for the win condition checker
	players := make([]rules.Player, len(e.gs.Players))
	for i := range e.gs.Players {
		players[i] = &e.gs.Players[i]
	}
	
	wasGameOver := e.gameOver
	gameOver, _ := e.winCondition.CheckGameOver(players)
	e.gameOver = gameOver
	
	if !wasGameOver && e.gameOver {
		l.Info().Msg("Game over condition met")
	} else if wasGameOver && !e.gameOver {
		// This would be highly unusual
		l.Error().Msg("Game was over, but is no longer over. This is unexpected!")
	}
}

// Public accessors
func (e *Engine) GameState() GameState { return *e.gs }
func (e *Engine) IsGameOver() bool   { return e.gameOver }

// GetWinner returns the winning player ID, or -1 if game isn't over or it's a draw
func (e *Engine) GetWinner() int {
	e.logger.Debug().Bool("is_game_over_state", e.gameOver).Msg("GetWinner called")
	if !e.gameOver {
		e.logger.Warn().Msg("GetWinner called but game is not actually over according to internal state.")
		return -1
	}
	
	// Create a slice of Player interfaces for the win condition checker
	players := make([]rules.Player, len(e.gs.Players))
	for i := range e.gs.Players {
		players[i] = &e.gs.Players[i]
	}
	
	_, winnerID := e.winCondition.CheckGameOver(players)
	return winnerID
}

// GetLegalActionMask returns a flattened boolean mask indicating which actions are legal for the given player.
// For a board of width W and height H:
// - Total actions = W * H * 4 (4 directions per tile)  
// - Index = (y * W + x) * 4 + direction
// - Directions: 0=up, 1=right, 2=down, 3=left
// - true = legal move, false = illegal move
func (e *Engine) GetLegalActionMask(playerID int) []bool {
	// If player is not alive or invalid, return empty mask
	if playerID < 0 || playerID >= len(e.gs.Players) {
		maskSize := e.gs.Board.W * e.gs.Board.H * 4
		return make([]bool, maskSize)
	}
	
	player := &e.gs.Players[playerID]
	return e.legalMoves.GetLegalActionMask(e.gs.Board, player, player.OwnedTiles)
}