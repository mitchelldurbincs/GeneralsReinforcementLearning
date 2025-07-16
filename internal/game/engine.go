package game

import (
	"context"
	"errors" 
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/mapgen"
	"github.com/rs/zerolog"
)

type Engine struct {
	gs       *GameState
	rng      *rand.Rand
	gameOver bool
	logger   zerolog.Logger
	
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

	engineLogger.Info().
		Int("width", cfg.Width).
		Int("height", cfg.Height).
		Int("num_players", cfg.Players).
		Msg("Initializing new game engine")

	if cfg.Rng == nil {
		seed := time.Now().UnixNano()
		engineLogger.Debug().Int64("seed", seed).Msg("RNG was nil, created new default RNG")
		cfg.Rng = rand.New(rand.NewSource(seed))
	} else {
		engineLogger.Debug().Msg("Using provided RNG")
	}

	config := mapgen.DefaultMapConfig(cfg.Width, cfg.Height, cfg.Players)
	generator := mapgen.NewGenerator(config, cfg.Rng)
	engineLogger.Debug().Interface("map_config", config).Msg("Map generator configured")
	board := generator.GenerateMap()
	engineLogger.Debug().Msg("Map generated")

	playerSlice := make([]Player, cfg.Players)
	for i := range cfg.Players {
		playerSlice[i] = Player{
			ID:         i,
			Alive:      true,
			GeneralIdx: -1,
			OwnedTiles: make([]int, 0, 50), // Pre-allocate for typical ownership
		}
	}
	engineLogger.Debug().Int("num_players", len(playerSlice)).Msg("Players initialized")

	e := &Engine{
		gs: &GameState{
			Board:                  board,
			Players:                playerSlice,
			ChangedTiles:           make(map[int]struct{}),
			FogOfWarEnabled:        true,
			VisibilityChangedTiles: make(map[int]struct{}),
		},
		rng:                 cfg.Rng,
		gameOver:            false,
		logger:              engineLogger,
		tempTileOwnership:   make(map[int]int),
		tempAffectedPlayers: make(map[int]struct{}),
	}

	e.updatePlayerStats() // This will use e.logger
	e.updateFogOfWar() // Initialize fog of war visibility
	e.logger.Info().Msg("Game engine created and initial stats updated")
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

// processActions handles all actions for this turn.
// It now accepts a context to check for cancellation during its loop.
func (e *Engine) processActions(ctx context.Context, actions []core.Action, l zerolog.Logger) error {
	l.Debug().Msg("Sorting actions for deterministic processing")
	sort.Slice(actions, func(i, j int) bool {
		return actions[i].GetPlayerID() < actions[j].GetPlayerID()
	})

	// Pre-allocate slice with reasonable capacity to reduce allocations
	allCaptureDetailsThisTurn := make([]core.CaptureDetails, 0, len(actions)/4)
	var encounteredError error

	for i, action := range actions {
		// Periodically check for context cancellation, especially if 'actions' can be very long.
		if i%10 == 0 { // Example: Check every 10 actions
			select {
			case <-ctx.Done():
				l.Warn().Err(ctx.Err()).Int("actions_processed_before_cancel", i).Msg("Action processing loop cancelled or timed out")
				return ctx.Err()
			default:
				// Continue processing
			}
		}

		playerID := action.GetPlayerID()

		if playerID < 0 || playerID >= len(e.gs.Players) || !e.gs.Players[playerID].Alive {
			l.Warn().Int("player_id", playerID).Interface("action", action).Msg("Skipping action from dead or invalid player")
			continue
		}

		l.Debug().Int("player_id", playerID).Interface("action", action).Msg("Applying action")
		switch act := action.(type) {
		case *core.MoveAction:
			// Assuming ApplyMoveAction itself doesn't need context for internal long computations.
			// If ApplyMoveAction also called action.Validate(..., logger), that logger could be actionLogger.
			// And if action.Validate took context, it would be passed there too.
			// For now, ApplyMoveAction doesn't take context in its signature.
			captureDetail, err := core.ApplyMoveAction(e.gs.Board, act, e.gs.ChangedTiles)
			if err != nil {
				l.Error().Err(err).
					Int("player_id", playerID).
					Interface("action_details", act).
					Msg("Failed to apply move action")
				if encounteredError == nil {
					encounteredError = err
				}
				continue
			}
			if captureDetail != nil {
				l.Debug().
					Int("player_id", playerID).
					Interface("capture_details", *captureDetail).
					Msg("Move resulted in capture")
				allCaptureDetailsThisTurn = append(allCaptureDetailsThisTurn, *captureDetail)
				// Mark tile for visibility update since ownership changed
				capturedTileIdx := e.gs.Board.Idx(captureDetail.X, captureDetail.Y)
				e.gs.VisibilityChangedTiles[capturedTileIdx] = struct{}{}
			}
		default:
			l.Warn().Int("player_id", playerID).Str("action_type", core.GetActionType(action)).Msg("Unhandled action type in processActions")
		}
	}

	if len(allCaptureDetailsThisTurn) > 0 {
		l.Debug().Int("num_captures_this_turn", len(allCaptureDetailsThisTurn)).Msg("Processing captures for eliminations")
		eliminationOrders := core.ProcessCaptures(allCaptureDetailsThisTurn)
		if len(eliminationOrders) > 0 {
			e.handleEliminationsAndTileTurnover(eliminationOrders, l)
		}
	}

	return encounteredError
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

// updatePlayerStats recalculates player statistics
// On turn 0 or when no previous stats exist, it does a full scan.
// Otherwise, it uses incremental updates based on ChangedTiles.
func (e *Engine) updatePlayerStats() {
	// Only do full update if we have changed tiles or on initialization
	if len(e.gs.ChangedTiles) == 0 && e.gs.Turn > 0 {
		// No changes, stats are already up to date
		e.logger.Debug().Msg("No tile changes, skipping player stats update")
		return
	}

	e.logger.Debug().Int("changed_tiles", len(e.gs.ChangedTiles)).Msg("Updating player stats")

	// On turn 0 or if we have too many changes, do a full update
	// Using 20% of board size as threshold for full update vs incremental
	fullUpdateThreshold := len(e.gs.Board.T) / 5
	if e.gs.Turn == 0 || len(e.gs.ChangedTiles) > fullUpdateThreshold {
		e.logger.Debug().Msg("Performing full stats update")
		e.performFullStatsUpdate()
		return
	}

	// Incremental update based on changed tiles
	e.logger.Debug().Msg("Performing incremental stats update")
	e.performIncrementalStatsUpdate()
}

// performFullStatsUpdate does a complete recalculation of all player stats
func (e *Engine) performFullStatsUpdate() {
	for pid := range e.gs.Players {
		e.gs.Players[pid].ArmyCount = 0
		e.gs.Players[pid].GeneralIdx = -1
		e.gs.Players[pid].OwnedTiles = e.gs.Players[pid].OwnedTiles[:0] // Clear but keep capacity
	}

	for i, t := range e.gs.Board.T {
		if t.IsNeutral() {
			continue
		}
		if t.Owner < 0 || t.Owner >= len(e.gs.Players) {
			e.logger.Error().Int("tile_index", i).Int("tile_owner", t.Owner).Int("num_players", len(e.gs.Players)).Msg("Invalid owner found on tile during player stats update")
			continue
		}

		p := &e.gs.Players[t.Owner]
		p.ArmyCount += t.Army
		p.OwnedTiles = append(p.OwnedTiles, i)

		if t.IsGeneral() {
			p.GeneralIdx = i
		}
	}
}

// performIncrementalStatsUpdate updates stats based only on changed tiles
func (e *Engine) performIncrementalStatsUpdate() {
	// We need to track the old state of tiles to properly update stats
	// Since we don't have that, we'll need to reconstruct ownership from OwnedTiles
	
	// Clear and reuse the temporary map instead of allocating new one
	for k := range e.tempTileOwnership {
		delete(e.tempTileOwnership, k)
	}
	
	// Build tile ownership map for quick lookup
	for pid := range e.gs.Players {
		for _, tileIdx := range e.gs.Players[pid].OwnedTiles {
			e.tempTileOwnership[tileIdx] = pid
		}
	}
	
	// Process each changed tile
	for tileIdx := range e.gs.ChangedTiles {
		if tileIdx < 0 || tileIdx >= len(e.gs.Board.T) {
			continue
		}
		
		tile := &e.gs.Board.T[tileIdx]
		oldOwner, hadOldOwner := e.tempTileOwnership[tileIdx]
		newOwner := tile.Owner
		
		// Remove from old owner's stats
		if hadOldOwner && oldOwner >= 0 && oldOwner < len(e.gs.Players) {
			p := &e.gs.Players[oldOwner]
			// Remove from OwnedTiles using swap-and-truncate for O(1) removal
			for i, idx := range p.OwnedTiles {
				if idx == tileIdx {
					// Swap with last element and truncate
					lastIdx := len(p.OwnedTiles) - 1
					p.OwnedTiles[i] = p.OwnedTiles[lastIdx]
					p.OwnedTiles = p.OwnedTiles[:lastIdx]
					break
				}
			}
			// Note: We can't accurately remove the old army count without knowing the previous value
			// This is a limitation of not tracking tile state changes
		}
		
		// Add to new owner's stats
		if newOwner >= 0 && newOwner < len(e.gs.Players) && !tile.IsNeutral() {
			p := &e.gs.Players[newOwner]
			// Add to OwnedTiles if not already there
			found := false
			for _, idx := range p.OwnedTiles {
				if idx == tileIdx {
					found = true
					break
				}
			}
			if !found {
				p.OwnedTiles = append(p.OwnedTiles, tileIdx)
			}
			
			if tile.IsGeneral() {
				p.GeneralIdx = tileIdx
			}
		}
	}
	
	// After updating ownership, recalculate army counts
	// We have to do this for all players since we don't track army deltas
	for pid := range e.gs.Players {
		e.gs.Players[pid].ArmyCount = 0
		for _, tileIdx := range e.gs.Players[pid].OwnedTiles {
			e.gs.Players[pid].ArmyCount += e.gs.Board.T[tileIdx].Army
		}
	}

	for pid := range e.gs.Players {
		oldAliveStatus := e.gs.Players[pid].Alive
		e.gs.Players[pid].Alive = e.gs.Players[pid].GeneralIdx != -1
		if oldAliveStatus && !e.gs.Players[pid].Alive {
			e.logger.Info().Int("player_id", pid).Msg("Player lost their general and is now marked as dead")
		} else if !oldAliveStatus && e.gs.Players[pid].Alive {
			// This case should ideally not happen unless a general is respawned/reassigned.
			e.logger.Warn().Int("player_id", pid).Msg("Player was dead and is now marked as alive - unexpected general appearance?")
		}
	}
	e.logger.Debug().Msg("Player stats updated")
}

func (e *Engine) updateFogOfWar() {
	if !e.gs.FogOfWarEnabled {
		return
	}

	// On turn 0 or if too many visibility changes, do full update
	fullUpdateThreshold := len(e.gs.Board.T) / 10 // 10% of board
	if e.gs.Turn == 0 || len(e.gs.VisibilityChangedTiles) > fullUpdateThreshold {
		e.performFullVisibilityUpdate()
		return
	}

	// Incremental update based on changed tiles
	e.performIncrementalVisibilityUpdate()
}

// performFullVisibilityUpdate recalculates all visibility from scratch
func (e *Engine) performFullVisibilityUpdate() {
	// Clear all visibility
	for i := range e.gs.Board.T {
		for pid := range e.gs.Players {
			e.gs.Board.T[i].Visible[pid] = false
		}
	}

	// Set visibility from all owned tiles
	for pid, p := range e.gs.Players {
		if !p.Alive {
			continue
		}
		for _, tileIdx := range p.OwnedTiles {
			e.setVisibilityAround(tileIdx, pid)
		}
	}
	
	e.logger.Debug().Msg("Performed full visibility update")
}

// performIncrementalVisibilityUpdate only updates visibility for changed tiles
func (e *Engine) performIncrementalVisibilityUpdate() {
	// For each tile that changed ownership, update visibility
	for tileIdx := range e.gs.VisibilityChangedTiles {
		if tileIdx < 0 || tileIdx >= len(e.gs.Board.T) {
			continue
		}
		
		// Get the current owner of the tile
		currentOwner := e.gs.Board.T[tileIdx].Owner
		
		// Clear visibility for all players around this tile first
		// This is necessary because we don't track previous owners
		e.clearVisibilityAround(tileIdx)
		
		// If tile has a new owner, grant visibility
		if currentOwner >= 0 && currentOwner < len(e.gs.Players) && e.gs.Players[currentOwner].Alive {
			e.setVisibilityAround(tileIdx, currentOwner)
		}
	}
	
	// Re-establish visibility from all owned tiles of affected players
	// This prevents removing visibility that should still exist from other tiles
	
	// Clear and reuse temporary map
	for k := range e.tempAffectedPlayers {
		delete(e.tempAffectedPlayers, k)
	}
	
	for tileIdx := range e.gs.VisibilityChangedTiles {
		tile := &e.gs.Board.T[tileIdx]
		if tile.Owner >= 0 {
			e.tempAffectedPlayers[tile.Owner] = struct{}{}
		}
	}
	
	for pid := range e.tempAffectedPlayers {
		if !e.gs.Players[pid].Alive {
			continue
		}
		for _, ownedTileIdx := range e.gs.Players[pid].OwnedTiles {
			e.setVisibilityAround(ownedTileIdx, pid)
		}
	}
	
	e.logger.Debug().Int("changed_tiles", len(e.gs.VisibilityChangedTiles)).Msg("Performed incremental visibility update")
}

// setVisibilityAround sets visibility for a player in a 3x3 area around a tile
func (e *Engine) setVisibilityAround(tileIdx int, playerID int) {
	x, y := e.gs.Board.XY(tileIdx)
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			nx, ny := x+dx, y+dy
			if e.gs.Board.InBounds(nx, ny) {
				e.gs.Board.T[e.gs.Board.Idx(nx, ny)].Visible[playerID] = true
			}
		}
	}
}

// clearVisibilityAround clears visibility for all players in a 3x3 area around a tile
func (e *Engine) clearVisibilityAround(tileIdx int) {
	x, y := e.gs.Board.XY(tileIdx)
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			nx, ny := x+dx, y+dy
			if e.gs.Board.InBounds(nx, ny) {
				idx := e.gs.Board.Idx(nx, ny)
				for pid := range e.gs.Players {
					e.gs.Board.T[idx].Visible[pid] = false
				}
			}
		}
	}
}


// checkGameOver determines if the game has ended
func (e *Engine) checkGameOver(l zerolog.Logger) {
	l.Debug().Msg("Checking game over conditions")
	aliveCount := 0
	var alivePlayers []int
	for _, p := range e.gs.Players {
		if p.Alive {
			aliveCount++
			alivePlayers = append(alivePlayers, p.ID)
		}
	}

	wasGameOver := e.gameOver
	// Game is over only if:
	// - 0 players alive (draw)
	// - 1 player alive AND there were originally more than 1 player
	if len(e.gs.Players) > 1 {
		e.gameOver = aliveCount <= 1
	} else {
		e.gameOver = aliveCount == 0
	}

	if !wasGameOver && e.gameOver {
		l.Info().Int("alive_player_count", aliveCount).Interface("alive_players_ids", alivePlayers).Msg("Game over condition met")
	} else if wasGameOver && !e.gameOver {
		// This would be highly unusual
		l.Error().Int("alive_player_count", aliveCount).Msg("Game was over, but is no longer over. This is unexpected!")
	}
	l.Debug().Bool("is_game_over", e.gameOver).Int("alive_player_count", aliveCount).Msg("Game over check complete")
}

// Public accessors
func (e *Engine) GameState() GameState { return *e.gs }
func (e *Engine) IsGameOver() bool   { return e.gameOver }

// GetWinner returns the winning player ID, or -1 if game isn't over or it's a draw
func (e *Engine) GetWinner() int {
	e.logger.Debug().Bool("is_game_over_state", e.gameOver).Msg("GetWinner called")
	if !e.gameOver {
		e.logger.Warn().Msg("GetWinner called but game is not actually over according to internal state.")
		// It's debatable whether to return -1 or re-check. For now, trust e.gameOver.
		// If there's a possibility of e.gameOver being stale, one might call checkGameOver here.
		return -1
	}
	for _, p := range e.gs.Players {
		if p.Alive {
			e.logger.Info().Int("winner_player_id", p.ID).Msg("Winner determined")
			return p.ID
		}
	}
	e.logger.Info().Msg("No winner found (draw, or all players eliminated simultaneously)")
	return -1 // Draw or no winner if all eliminated somehow
}

// ANSI color codes (Unchanged - for Board rendering)
const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
	ColorWhite  = "\033[37m"
	ColorGray   = "\033[90m"

	BgRed    = "\033[41m"
	BgGreen  = "\033[42m"
	BgYellow = "\033[43m"
	BgBlue   = "\033[44m"
	BgPurple = "\033[45m"
	BgCyan   = "\033[46m"
)

var playerColors = []string{ColorRed, ColorBlue, ColorGreen, ColorYellow, ColorPurple, ColorCyan}

// Board returns a string representation of the board
func (e *Engine) Board(playerID int) string {
	const (
		EmptySymbol    = "·"
		CitySymbol     = "⬢"
		GeneralSymbol  = "♔"
		MountainSymbol = "▲"
	)
	
	// Pre-allocate buffer size based on board dimensions
	// Each cell takes roughly: 2 chars for symbol + ~20 chars for ANSI codes
	// Plus headers and legend
	width := e.gs.Board.W
	height := e.gs.Board.H
	estimatedSize := (width*22 + 10) * (height + 3) + 100 // Extra for headers and legend
	
	var sb strings.Builder
	sb.Grow(estimatedSize)
	
	// Header row
	sb.WriteString("    ")
	for x := 0; x < width; x++ {
		sb.WriteString(core.IntToStringFixedWidth(x, 2))
	}
	sb.WriteString("\n")
	
	// Board rows
	for y := 0; y < height; y++ {
		sb.WriteString(core.IntToStringFixedWidth(y, 2))
		sb.WriteString(" ")
		for x := 0; x < width; x++ {
			t := e.gs.Board.T[e.gs.Board.Idx(x, y)]
			e.getTileDisplayDirect(&sb, t, playerID)
		}
		sb.WriteString("\n")
	}
	
	// Legend
	sb.WriteString("\n")
	sb.WriteString(EmptySymbol)
	sb.WriteString("=empty ")
	sb.WriteString(CitySymbol)
	sb.WriteString("=city ")
	sb.WriteString(GeneralSymbol)
	sb.WriteString("=general ")
	sb.WriteString(MountainSymbol)
	sb.WriteString("=mountain A-H=players\n")
	
	return sb.String()
}

// getTileDisplay writes the tile display directly to the strings.Builder to avoid allocations
func (e *Engine) getTileDisplayDirect(sb *strings.Builder, t core.Tile, playerID int) {
	const (
		EmptySymbol    = "·"
		CitySymbol     = "⬢"
		GeneralSymbol  = "♔"
		MountainSymbol = "▲"
		PlayerSymbols  = "ABCDEFGH"
	)

	visible := playerID < 0 || !e.gs.FogOfWarEnabled || t.Visible[playerID]

	// Write color first
	if !visible {
		sb.WriteString(ColorGray)
		sb.WriteString(" ")
	} else if t.IsMountain() {
		sb.WriteString(ColorGray)
		sb.WriteString(" ")
		sb.WriteString(MountainSymbol)
	} else if t.IsGeneral() {
		sb.WriteString(getPlayerColor(t.Owner))
		sb.WriteByte(PlayerSymbols[t.Owner%len(PlayerSymbols)])
		sb.WriteString(GeneralSymbol)
	} else if t.IsCity() && t.IsNeutral() {
		sb.WriteString(ColorWhite)
		sb.WriteString(" ")
		sb.WriteString(CitySymbol)
	} else if t.IsCity() {
		sb.WriteString(getPlayerColor(t.Owner))
		sb.WriteByte(PlayerSymbols[t.Owner%len(PlayerSymbols)])
		sb.WriteString(CitySymbol)
	} else if t.IsNeutral() && t.Type == core.TileNormal {
		sb.WriteString(ColorGray)
		if t.Army == 0 {
			sb.WriteString(" ")
			sb.WriteString(EmptySymbol)
		} else if t.Army >= 100 {
			sb.WriteString("++")
		} else if t.Army >= 10 {
			sb.WriteString(core.IntToStringFixedWidth(t.Army, 2))
		} else {
			sb.WriteString(" ")
			sb.WriteString(core.IntToStringFixedWidth(t.Army, 1))
		}
	} else { // Player-owned normal tiles or other neutral tiles
		if t.IsNeutral() {
			sb.WriteString(ColorGray)
			if t.Army >= 100 {
				sb.WriteString("++")
			} else {
				sb.WriteString(core.IntToStringFixedWidth(t.Army, 2))
			}
		} else {
			sb.WriteString(getPlayerColor(t.Owner))
			sb.WriteByte(PlayerSymbols[t.Owner%len(PlayerSymbols)])
			if t.Army < 10 {
				sb.WriteString(core.IntToStringFixedWidth(t.Army, 1))
			} else {
				sb.WriteString("+") // P+ for armies >=10
			}
		}
	}
	sb.WriteString(ColorReset)
}

func (e *Engine) getTileDisplay(t core.Tile, playerID int) (string, string) {
	const (
		EmptySymbol    = "·"
		CitySymbol     = "⬢"
		GeneralSymbol  = "♔"
		MountainSymbol = "▲"
		PlayerSymbols  = "ABCDEFGH"
	)

	visible := playerID < 0 || !e.gs.FogOfWarEnabled || t.Visible[playerID]

	var symbol string
	var color string
	switch {
	case !visible:
		symbol = " "
		color = ColorGray
	case t.IsMountain():
		symbol = " " + MountainSymbol
		color = ColorGray
	case t.IsGeneral():
		playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
		symbol = playerChar + GeneralSymbol
		color = getPlayerColor(t.Owner)
	case t.IsCity() && t.IsNeutral():
		symbol = " " + CitySymbol
		color = ColorWhite
	case t.IsCity():
		playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
		symbol = playerChar + CitySymbol
		color = getPlayerColor(t.Owner)
	case t.IsNeutral() && t.Type == core.TileNormal:
		if t.Army == 0 {
			symbol = " " + EmptySymbol
		} else {
			symbol = core.IntToStringFixedWidth(t.Army, 2) // Assuming core.IntToStringFixedWidth
			if t.Army >= 100 { // Adjusted for 2 chars
				symbol = "++"
			} else if t.Army >= 10 {
				// Handled by IntToStringFixedWidth up to 99
			} else {
				// prepend space if single digit
				symbol = " " + core.IntToStringFixedWidth(t.Army, 1)
			}
		}
		color = ColorGray
	default: // Player-owned normal tiles or other neutral tiles
		if t.IsNeutral() {
			symbol = core.IntToStringFixedWidth(t.Army, 2)
			if t.Army >= 100 {
				symbol = "++"
			}
			color = ColorGray
		} else {
			playerChar := string(PlayerSymbols[t.Owner%len(PlayerSymbols)])
			if t.Army < 10 {
				symbol = playerChar + core.IntToStringFixedWidth(t.Army, 1)
			} else {
				symbol = playerChar + "+" // P+ for armies >=10
			}
			color = getPlayerColor(t.Owner)
		}
	}
	return symbol, color
}

func getPlayerColor(playerID int) string {
	if playerID >= 0 && playerID < len(playerColors) {
		return playerColors[playerID]
	}
	return ColorWhite
}