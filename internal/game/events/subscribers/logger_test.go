package subscribers_test

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events/subscribers"
)

func TestLoggerSubscriber(t *testing.T) {
	// Create a buffer to capture log output
	var buf bytes.Buffer
	logger := zerolog.New(&buf).With().Timestamp().Logger()

	// Create logger subscriber
	logSub := subscribers.NewLoggerSubscriber("test-logger", logger, zerolog.InfoLevel)

	// Test ID
	assert.Equal(t, "test-logger", logSub.ID())

	// Test InterestedIn - should be interested in all events by default
	assert.True(t, logSub.InterestedIn(events.TypeGameStarted))
	assert.True(t, logSub.InterestedIn(events.TypeTurnStarted))
	assert.True(t, logSub.InterestedIn("any.event.type"))
}

func TestLoggerSubscriberEventLogging(t *testing.T) {
	var buf bytes.Buffer
	logger := zerolog.New(&buf)

	logSub := subscribers.NewLoggerSubscriber("event-logger", logger, zerolog.InfoLevel)

	// Create and handle various events
	testCases := []struct {
		name  string
		event events.Event
		check func(t *testing.T, logLine map[string]interface{})
	}{
		{
			name: "GameStartedEvent",
			event: &events.GameStartedEvent{
				BaseEvent: events.BaseEvent{
					EventType: events.TypeGameStarted,
					Time:      time.Now(),
					Game:      "test-game-1",
				},
				NumPlayers: 4,
				MapWidth:   20,
				MapHeight:  20,
			},
			check: func(t *testing.T, logLine map[string]interface{}) {
				assert.Equal(t, "Game event", logLine["message"])
				assert.Equal(t, float64(4), logLine["num_players"])
				assert.Equal(t, float64(20), logLine["map_width"])
				assert.Equal(t, float64(20), logLine["map_height"])
			},
		},
		{
			name: "TurnStartedEvent",
			event: &events.TurnStartedEvent{
				BaseEvent: events.BaseEvent{
					EventType: events.TypeTurnStarted,
					Time:      time.Now(),
					Game:      "test-game-1",
				},
				TurnNumber: 5,
			},
			check: func(t *testing.T, logLine map[string]interface{}) {
				assert.Equal(t, "Game event", logLine["message"])
				assert.Equal(t, float64(5), logLine["turn"])
			},
		},
		{
			name: "CombatResolvedEvent",
			event: &events.CombatResolvedEvent{
				BaseEvent: events.BaseEvent{
					EventType: events.TypeCombatResolved,
					Time:      time.Now(),
					Game:      "test-game-1",
				},
				AttackerID:     0,
				DefenderID:     1,
				AttackerLosses: 10,
				DefenderLosses: 15,
				TileCaptured:   true,
			},
			check: func(t *testing.T, logLine map[string]interface{}) {
				assert.Equal(t, "Game event", logLine["message"])
				assert.Equal(t, float64(0), logLine["attacker_id"])
				assert.Equal(t, float64(1), logLine["defender_id"])
				assert.Equal(t, float64(10), logLine["attacker_losses"])
				assert.Equal(t, float64(15), logLine["defender_losses"])
				assert.Equal(t, true, logLine["tile_captured"])
			},
		},
		{
			name: "PlayerEliminatedEvent",
			event: &events.PlayerEliminatedEvent{
				BaseEvent: events.BaseEvent{
					EventType: events.TypePlayerEliminated,
					Time:      time.Now(),
					Game:      "test-game-1",
				},
				PlayerID:     2,
				EliminatedBy: 0,
				FinalRank:    3,
			},
			check: func(t *testing.T, logLine map[string]interface{}) {
				assert.Equal(t, "Game event", logLine["message"])
				assert.Equal(t, float64(2), logLine["player_id"])
				assert.Equal(t, float64(0), logLine["eliminated_by"])
				assert.Equal(t, float64(3), logLine["final_rank"])
			},
		},
		{
			name: "GameEndedEvent",
			event: &events.GameEndedEvent{
				BaseEvent: events.BaseEvent{
					EventType: events.TypeGameEnded,
					Time:      time.Now(),
					Game:      "test-game-1",
				},
				Winner:   0,
				Duration: time.Minute * 5,
			},
			check: func(t *testing.T, logLine map[string]interface{}) {
				assert.Equal(t, "Game event", logLine["message"])
				assert.Equal(t, float64(0), logLine["winner"])
				assert.Equal(t, float64(300000), logLine["duration"]) // 5 minutes in ms
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			buf.Reset()
			logSub.HandleEvent(tc.event)

			// Parse the log output
			logOutput := buf.String()
			require.NotEmpty(t, logOutput, "Log output should not be empty")

			var logLine map[string]interface{}
			err := json.Unmarshal([]byte(logOutput), &logLine)
			require.NoError(t, err, "Should be able to parse log output as JSON")

			// Common checks
			assert.Equal(t, "info", logLine["level"])
			assert.Equal(t, tc.event.Type(), logLine["event_type"])
			assert.Equal(t, "test-game-1", logLine["game_id"])

			// Event-specific checks
			tc.check(t, logLine)
		})
	}
}

func TestLoggerSubscriberWithFilter(t *testing.T) {
	var buf bytes.Buffer
	logger := zerolog.New(&buf)

	// Create logger with filter
	logSub := subscribers.NewLoggerSubscriber("filtered-logger", logger, zerolog.InfoLevel)
	logSub.SetEventFilter([]string{events.TypeGameStarted, events.TypeGameEnded})

	// Should be interested only in filtered events
	assert.True(t, logSub.InterestedIn(events.TypeGameStarted))
	assert.True(t, logSub.InterestedIn(events.TypeGameEnded))
	assert.False(t, logSub.InterestedIn(events.TypeTurnStarted))
	assert.False(t, logSub.InterestedIn(events.TypeMoveExecuted))

	// Test that only filtered events are logged
	events := []events.Event{
		events.NewGameStartedEvent("game1", 2, 10, 10),
		events.NewTurnStartedEvent("game1", 1), // Should not be logged
		&events.GameEndedEvent{
			BaseEvent: events.BaseEvent{
				EventType: events.TypeGameEnded,
				Time:      time.Now(),
				Game:      "game1",
			},
			Winner: 0,
		},
	}

	for _, event := range events {
		buf.Reset()
		if logSub.InterestedIn(event.Type()) {
			logSub.HandleEvent(event)
			assert.NotEmpty(t, buf.String(), "Should log event of type %s", event.Type())
		} else {
			// The event bus won't call HandleEvent for events the subscriber isn't interested in
			// So we shouldn't test handling events we're not interested in
		}
	}
}

func TestLoggerSubscriberLogLevels(t *testing.T) {
	// Test that logger uses the configured log level
	testCases := []struct {
		name     string
		logLevel zerolog.Level
		expected string
	}{
		{"Debug", zerolog.DebugLevel, "debug"},
		{"Info", zerolog.InfoLevel, "info"},
		{"Warn", zerolog.WarnLevel, "warn"},
		{"Error", zerolog.ErrorLevel, "error"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var buf bytes.Buffer
			logger := zerolog.New(&buf).Level(tc.logLevel)
			
			logSub := subscribers.NewLoggerSubscriber("level-logger", logger, tc.logLevel)
			
			// Create an event
			event := events.NewGameStartedEvent("game1", 2, 10, 10)
			logSub.HandleEvent(event)
			
			// The subscriber will always log events at the configured level
			// The actual output depends on the logger's level filter
			if buf.Len() > 0 {
				var logLine map[string]interface{}
				err := json.Unmarshal(buf.Bytes(), &logLine)
				require.NoError(t, err)
				
				// The log level should match what we configured for the subscriber
				assert.Equal(t, tc.expected, logLine["level"])
			}
		})
	}
}

func TestLoggerSubscriberDevelopmentMode(t *testing.T) {
	var buf bytes.Buffer
	logger := zerolog.New(&buf)

	logSub := subscribers.NewLoggerSubscriber("dev-logger", logger, zerolog.InfoLevel)
	logSub.SetDevMode(true)

	// In development mode, should log additional details
	event := &events.MoveExecutedEvent{
		BaseEvent: events.BaseEvent{
			EventType: events.TypeMoveExecuted,
			Time:      time.Now(),
			Game:      "dev-game",
		},
		PlayerID:    0,
		From:        core.NewCoordinate(5, 5),
		To:          core.NewCoordinate(6, 5),
		ArmiesMoved: 10,
	}

	logSub.HandleEvent(event)

	logOutput := buf.String()
	require.NotEmpty(t, logOutput)

	// In development mode, we expect event_data field
	assert.Contains(t, logOutput, "event_data")
	
	var logLine map[string]interface{}
	err := json.Unmarshal(buf.Bytes(), &logLine)
	require.NoError(t, err)
	
	// event_data might be a map or object, not a string when parsed as JSON
	eventData, ok := logLine["event_data"]
	require.True(t, ok, "event_data should be present")
	
	// Convert back to string to check contents
	eventDataBytes, err := json.Marshal(eventData)
	require.NoError(t, err)
	eventDataStr := string(eventDataBytes)
	
	// The event data should contain information about the event
	assert.Contains(t, eventDataStr, "move.executed")
	assert.Contains(t, eventDataStr, "PlayerID")
}

func TestLoggerSubscriberBenchmark(t *testing.T) {
	// This is not a real benchmark, just a test to ensure performance is reasonable
	var buf bytes.Buffer
	logger := zerolog.New(&buf).Level(zerolog.Disabled) // Disable actual logging for speed

	logSub := subscribers.NewLoggerSubscriber("bench-logger", logger, zerolog.InfoLevel)

	start := time.Now()
	numEvents := 10000

	for i := 0; i < numEvents; i++ {
		event := events.NewTurnStartedEvent("bench-game", i)
		logSub.HandleEvent(event)
	}

	elapsed := time.Since(start)
	eventsPerSecond := float64(numEvents) / elapsed.Seconds()

	// Should be able to process at least 100k events per second with logging disabled
	assert.Greater(t, eventsPerSecond, 100000.0, 
		"Logger should process at least 100k events/sec, got %.0f", eventsPerSecond)
}