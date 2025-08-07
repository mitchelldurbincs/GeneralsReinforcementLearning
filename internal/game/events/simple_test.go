package events_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/events"
)

// TestSubscriber implements the Subscriber interface for testing
type TestSubscriber struct {
	id         string
	events     []events.Event
	interested map[string]bool
}

func NewTestSubscriber(id string, interestedTypes ...string) *TestSubscriber {
	interested := make(map[string]bool)
	for _, t := range interestedTypes {
		interested[t] = true
	}
	return &TestSubscriber{
		id:         id,
		interested: interested,
	}
}

func (ts *TestSubscriber) ID() string {
	return ts.id
}

func (ts *TestSubscriber) HandleEvent(event events.Event) {
	ts.events = append(ts.events, event)
}

func (ts *TestSubscriber) InterestedIn(eventType string) bool {
	if len(ts.interested) == 0 {
		return true // Interested in all events if not specified
	}
	return ts.interested[eventType]
}

func TestEventBusBasicFunctionality(t *testing.T) {
	bus := events.NewEventBus()

	// Create a test subscriber interested in game events
	subscriber := NewTestSubscriber("test1", events.TypeGameStarted, events.TypeGameEnded)
	bus.Subscribe(subscriber)

	// Publish a game started event
	gameStarted := events.NewGameStartedEvent("game1", 2, 10, 10)
	bus.Publish(gameStarted)

	// Verify subscriber received the event
	require.Len(t, subscriber.events, 1)
	assert.Equal(t, events.TypeGameStarted, subscriber.events[0].Type())
	assert.Equal(t, "game1", subscriber.events[0].GameID())

	// Publish a turn started event (subscriber not interested)
	turnStarted := events.NewTurnStartedEvent("game1", 1)
	bus.Publish(turnStarted)

	// Verify subscriber didn't receive it
	assert.Len(t, subscriber.events, 1)

	// Publish a game ended event
	gameEnded := &events.GameEndedEvent{
		BaseEvent: events.BaseEvent{
			EventType: events.TypeGameEnded,
			Time:      time.Now(),
			Game:      "game1",
		},
		Winner:   0,
		Duration: time.Minute,
	}
	bus.Publish(gameEnded)

	// Verify subscriber received it
	require.Len(t, subscriber.events, 2)
	assert.Equal(t, events.TypeGameEnded, subscriber.events[1].Type())
}

func TestEventBusUnsubscribe(t *testing.T) {
	bus := events.NewEventBus()

	subscriber := NewTestSubscriber("test2")
	bus.Subscribe(subscriber)

	// Publish event
	event := events.NewTurnStartedEvent("game2", 5)
	bus.Publish(event)

	// Verify received
	assert.Len(t, subscriber.events, 1)

	// Unsubscribe
	bus.Unsubscribe(subscriber.ID())

	// Publish another event
	event2 := events.NewTurnEndedEvent("game2", 6, 2, time.Millisecond*100)
	bus.Publish(event2)

	// Verify not received
	assert.Len(t, subscriber.events, 1)
}

func TestEventBusFunctionHandlers(t *testing.T) {
	bus := events.NewEventBus()

	received := []events.Event{}

	// Subscribe with function
	bus.SubscribeFunc(events.TypeMoveExecuted, func(e events.Event) {
		received = append(received, e)
	})

	// Publish move event
	moveEvent := &events.MoveExecutedEvent{
		BaseEvent: events.BaseEvent{
			EventType: events.TypeMoveExecuted,
			Time:      time.Now(),
			Game:      "game3",
		},
		PlayerID:    0,
		From:        core.NewCoordinate(5, 5),
		To:          core.NewCoordinate(6, 5),
		ArmiesMoved: 10,
	}
	bus.Publish(moveEvent)

	// Verify function was called
	require.Len(t, received, 1)
	assert.Equal(t, events.TypeMoveExecuted, received[0].Type())
}

func TestEventBusMultipleSubscribers(t *testing.T) {
	bus := events.NewEventBus()

	// Create multiple subscribers
	sub1 := NewTestSubscriber("sub1", events.TypeCombatResolved)
	sub2 := NewTestSubscriber("sub2", events.TypeCombatResolved)
	sub3 := NewTestSubscriber("sub3") // Interested in all

	bus.Subscribe(sub1)
	bus.Subscribe(sub2)
	bus.Subscribe(sub3)

	// Also add a function handler
	funcCalled := false
	bus.SubscribeFunc(events.TypeCombatResolved, func(e events.Event) {
		funcCalled = true
	})

	// Publish combat event
	combatEvent := &events.CombatResolvedEvent{
		BaseEvent: events.BaseEvent{
			EventType: events.TypeCombatResolved,
			Time:      time.Now(),
			Game:      "game4",
		},
		AttackerID:     0,
		DefenderID:     1,
		AttackerLosses: 5,
		DefenderLosses: 3,
		TileCaptured:   true,
	}
	bus.Publish(combatEvent)

	// Verify all subscribers received it
	assert.Len(t, sub1.events, 1)
	assert.Len(t, sub2.events, 1)
	assert.Len(t, sub3.events, 1)
	assert.True(t, funcCalled)
}

func TestEventBusPanicRecovery(t *testing.T) {
	bus := events.NewEventBus()

	// Add a handler that panics
	bus.SubscribeFunc(events.TypePlayerEliminated, func(e events.Event) {
		panic("test panic")
	})

	// Add a normal subscriber
	normalSub := NewTestSubscriber("normal")
	bus.Subscribe(normalSub)

	// Publish event - should not panic
	elimEvent := events.NewPlayerEliminatedEvent("game5", 1, 0, 2, 10)

	// This should not panic the whole program
	assert.NotPanics(t, func() {
		bus.Publish(elimEvent)
	})

	// Normal subscriber should still receive the event
	assert.Len(t, normalSub.events, 1)
}

func TestEventTimestamps(t *testing.T) {
	startTime := time.Now()

	// Create various events
	events := []events.Event{
		events.NewGameStartedEvent("game6", 4, 20, 20),
		events.NewTurnStartedEvent("game6", 1),
		events.NewTurnEndedEvent("game6", 1, 2, time.Millisecond*50),
		events.NewProductionAppliedEvent("game6", 10, 2, 2, 1),
	}

	// Verify timestamps
	for _, event := range events {
		assert.False(t, event.Timestamp().IsZero())
		assert.True(t, event.Timestamp().After(startTime) || event.Timestamp().Equal(startTime))
		assert.True(t, event.Timestamp().Before(time.Now().Add(time.Second)))
		assert.Equal(t, "game6", event.GameID())
	}
}

func TestEventMetadata(t *testing.T) {
	// Test events with metadata
	metadata := events.EventMetadata{
		PlayerID: 1,
		Turn:     5,
		Extra: map[string]interface{}{
			"custom_field": "value",
			"number":       42,
		},
	}

	actionProcessed := &events.ActionProcessedEvent{
		BaseEvent: events.BaseEvent{
			EventType: events.TypeActionProcessed,
			Time:      time.Now(),
			Game:      "game7",
		},
		Metadata: metadata,
		PlayerID: 1,
		Result:   "success",
	}

	// Verify metadata is preserved
	assert.Equal(t, 1, actionProcessed.Metadata.PlayerID)
	assert.Equal(t, 5, actionProcessed.Metadata.Turn)
	assert.Equal(t, "value", actionProcessed.Metadata.Extra["custom_field"])
	assert.Equal(t, 42, actionProcessed.Metadata.Extra["number"])
}

// Benchmark tests
func BenchmarkEventBusPublish(b *testing.B) {
	bus := events.NewEventBus()
	subscriber := NewTestSubscriber("bench")
	bus.Subscribe(subscriber)

	event := events.NewTurnStartedEvent("bench-game", 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.Publish(event)
	}
}

func BenchmarkEventBusMultipleSubscribers(b *testing.B) {
	bus := events.NewEventBus()

	// Add 10 subscribers
	for i := 0; i < 10; i++ {
		bus.Subscribe(NewTestSubscriber(string(rune('a' + i))))
	}

	event := events.NewTurnStartedEvent("bench-game", 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.Publish(event)
	}
}
