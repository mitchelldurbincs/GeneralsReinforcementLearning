package subscribers

import (
	"encoding/json"
	"fmt"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/rs/zerolog"
)

// LoggerSubscriber logs events to structured logs
type LoggerSubscriber struct {
	id              string
	logger          zerolog.Logger
	logLevel        zerolog.Level
	eventTypeFilter map[string]bool // If non-nil, only log these event types
	devMode         bool            // If true, log full event details
}

// NewLoggerSubscriber creates a new logger subscriber
func NewLoggerSubscriber(id string, logger zerolog.Logger, logLevel zerolog.Level) *LoggerSubscriber {
	return &LoggerSubscriber{
		id:       id,
		logger:   logger.With().Str("subscriber", "event_logger").Logger(),
		logLevel: logLevel,
	}
}

// ID returns the subscriber's unique identifier
func (ls *LoggerSubscriber) ID() string {
	return ls.id
}

// SetEventFilter sets which event types to log (nil means log all)
func (ls *LoggerSubscriber) SetEventFilter(eventTypes []string) {
	if len(eventTypes) == 0 {
		ls.eventTypeFilter = nil
		return
	}

	ls.eventTypeFilter = make(map[string]bool)
	for _, eventType := range eventTypes {
		ls.eventTypeFilter[eventType] = true
	}
}

// SetDevMode enables or disables development mode logging
func (ls *LoggerSubscriber) SetDevMode(enabled bool) {
	ls.devMode = enabled
}

// InterestedIn returns true if the subscriber wants to receive this event type
func (ls *LoggerSubscriber) InterestedIn(eventType string) bool {
	// If no filter is set, interested in all events
	if ls.eventTypeFilter == nil {
		return true
	}
	return ls.eventTypeFilter[eventType]
}

// HandleEvent processes an event by logging it
func (ls *LoggerSubscriber) HandleEvent(event events.Event) {
	eventLogger := ls.logger.With().
		Str("event_type", event.Type()).
		Str("game_id", event.GameID()).
		Time("timestamp", event.Timestamp()).
		Logger()

	// Create the base event log
	var logEvent *zerolog.Event
	switch ls.logLevel {
	case zerolog.DebugLevel:
		logEvent = eventLogger.Debug()
	case zerolog.InfoLevel:
		logEvent = eventLogger.Info()
	case zerolog.WarnLevel:
		logEvent = eventLogger.Warn()
	case zerolog.ErrorLevel:
		logEvent = eventLogger.Error()
	default:
		logEvent = eventLogger.Info()
	}

	// Add event-specific fields based on type
	switch e := event.(type) {
	case *events.GameStartedEvent:
		logEvent.
			Int("num_players", e.NumPlayers).
			Int("map_width", e.MapWidth).
			Int("map_height", e.MapHeight)

	case *events.GameEndedEvent:
		logEvent.
			Int("winner", e.Winner).
			Dur("duration", e.Duration).
			Int("final_turn", e.FinalTurn)

	case *events.TurnStartedEvent:
		logEvent.Int("turn", e.TurnNumber)

	case *events.TurnEndedEvent:
		logEvent.
			Int("turn", e.TurnNumber).
			Int("actions_count", e.ActionsCount).
			Dur("process_time", e.ProcessedTime)

	case *events.ActionSubmittedEvent:
		logEvent.
			Int("player_id", e.PlayerID).
			Str("action_type", fmt.Sprintf("%v", e.Action.GetType()))

	case *events.ActionProcessedEvent:
		logEvent.
			Int("player_id", e.PlayerID).
			Str("action_type", fmt.Sprintf("%v", e.Action.GetType())).
			Str("result", e.Result)

	case *events.MoveExecutedEvent:
		logEvent.
			Int("player_id", e.PlayerID).
			Int("from_x", e.From.X).
			Int("from_y", e.From.Y).
			Int("to_x", e.To.X).
			Int("to_y", e.To.Y).
			Int("armies_moved", e.ArmiesMoved).
			Bool("half", e.Half)

	case *events.CombatResolvedEvent:
		logEvent.
			Int("attacker_id", e.AttackerID).
			Int("defender_id", e.DefenderID).
			Int("location_x", e.Location.X).
			Int("location_y", e.Location.Y).
			Int("attacker_armies", e.AttackerArmies).
			Int("defender_armies", e.DefenderArmies).
			Int("attacker_losses", e.AttackerLosses).
			Int("defender_losses", e.DefenderLosses).
			Bool("tile_captured", e.TileCaptured)

	case *events.PlayerEliminatedEvent:
		logEvent.
			Int("player_id", e.PlayerID).
			Int("eliminated_by", e.EliminatedBy).
			Int("final_rank", e.FinalRank)

	case *events.ProductionAppliedEvent:
		logEvent.
			Int("tiles_produced", e.TilesProduced).
			Int("cities_produced", e.CitiesProduced).
			Int("generals_produced", e.GeneralsProduced)
	}

	// In dev mode, also log the full event as JSON
	if ls.devMode {
		if jsonData, err := json.Marshal(event); err == nil {
			logEvent.RawJSON("event_data", jsonData)
		}
	}

	// Send the log
	logEvent.Msg("Game event")
}
