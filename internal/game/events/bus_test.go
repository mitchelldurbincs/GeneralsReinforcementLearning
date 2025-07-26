package events

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestEventBus(t *testing.T) {
	bus := NewEventBus()
	
	// Test function handler
	received := false
	var receivedEvent Event
	
	bus.SubscribeFunc(TypeGameStarted, func(e Event) {
		received = true
		receivedEvent = e
	})
	
	// Publish event
	event := NewGameStartedEvent("test-game", 4, 20, 20)
	bus.Publish(event)
	
	// Verify event was received
	assert.True(t, received, "Event handler should have been called")
	assert.NotNil(t, receivedEvent, "Event should have been received")
	assert.Equal(t, TypeGameStarted, receivedEvent.Type())
	assert.Equal(t, "test-game", receivedEvent.GameID())
}

func TestEventBusMultipleSubscribers(t *testing.T) {
	bus := NewEventBus()
	
	// Track which handlers were called
	handler1Called := false
	handler2Called := false
	
	bus.SubscribeFunc(TypeTurnStarted, func(e Event) {
		handler1Called = true
	})
	
	bus.SubscribeFunc(TypeTurnStarted, func(e Event) {
		handler2Called = true
	})
	
	// Publish event
	event := NewTurnStartedEvent("test-game", 1)
	bus.Publish(event)
	
	// Both handlers should be called
	assert.True(t, handler1Called, "Handler 1 should have been called")
	assert.True(t, handler2Called, "Handler 2 should have been called")
}

// TestSubscriber is a test implementation of Subscriber
type TestSubscriber struct {
	id              string
	interestedTypes map[string]bool
	receivedEvents  []Event
}

func (ts *TestSubscriber) ID() string {
	return ts.id
}

func (ts *TestSubscriber) HandleEvent(e Event) {
	ts.receivedEvents = append(ts.receivedEvents, e)
}

func (ts *TestSubscriber) InterestedIn(eventType string) bool {
	if ts.interestedTypes == nil {
		return true
	}
	return ts.interestedTypes[eventType]
}

func TestEventBusSubscriber(t *testing.T) {
	bus := NewEventBus()
	
	// Create a test subscriber interested in specific events
	subscriber := &TestSubscriber{
		id: "test-subscriber",
		interestedTypes: map[string]bool{
			TypeGameStarted: true,
			TypeGameEnded:   true,
		},
		receivedEvents: []Event{},
	}
	
	bus.Subscribe(subscriber)
	
	// Publish various events
	bus.Publish(NewGameStartedEvent("test-game", 2, 10, 10))
	bus.Publish(NewTurnStartedEvent("test-game", 1))
	bus.Publish(NewGameEndedEvent("test-game", 0, time.Minute, 100))
	
	// Should only receive GameStarted and GameEnded
	assert.Len(t, subscriber.receivedEvents, 2)
	assert.Equal(t, TypeGameStarted, subscriber.receivedEvents[0].Type())
	assert.Equal(t, TypeGameEnded, subscriber.receivedEvents[1].Type())
	
	// Test unsubscribe
	bus.Unsubscribe(subscriber.ID())
	bus.Publish(NewGameStartedEvent("test-game", 2, 10, 10))
	
	// Should still have only 2 events
	assert.Len(t, subscriber.receivedEvents, 2)
}