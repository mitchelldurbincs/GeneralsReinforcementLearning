package events

import (
	"time"
)

// Event is the base interface for all game events
type Event interface {
	// Type returns the event type as a string for filtering and logging
	Type() string
	// Timestamp returns when the event occurred
	Timestamp() time.Time
	// GameID returns the ID of the game this event belongs to
	GameID() string
}

// BaseEvent provides common fields for all events
type BaseEvent struct {
	EventType string    `json:"type"`
	Time      time.Time `json:"timestamp"`
	Game      string    `json:"game_id"`
}

// Type implements Event interface
func (e BaseEvent) Type() string {
	return e.EventType
}

// Timestamp implements Event interface
func (e BaseEvent) Timestamp() time.Time {
	return e.Time
}

// GameID implements Event interface
func (e BaseEvent) GameID() string {
	return e.Game
}

// EventHandler is a function that processes events
type EventHandler func(Event)

// Subscriber represents an entity that can receive events
type Subscriber interface {
	// ID returns a unique identifier for this subscriber
	ID() string
	// HandleEvent processes an event
	HandleEvent(Event)
	// InterestedIn returns true if the subscriber wants to receive this event type
	InterestedIn(eventType string) bool
}

// EventMetadata contains additional context for events
type EventMetadata struct {
	// PlayerID associated with the event (if applicable)
	PlayerID int `json:"player_id,omitempty"`
	// Turn number when the event occurred
	Turn int `json:"turn,omitempty"`
	// Additional key-value pairs for context
	Extra map[string]interface{} `json:"extra,omitempty"`
}

// Publisher is the interface for publishing events
type Publisher interface {
	// Publish sends an event to all interested subscribers
	Publish(Event)
}

// Bus is the main event bus interface
type Bus interface {
	Publisher
	// Subscribe adds a new subscriber to the event bus
	Subscribe(Subscriber)
	// Unsubscribe removes a subscriber from the event bus
	Unsubscribe(subscriberID string)
	// SubscribeFunc adds a function handler for specific event types
	SubscribeFunc(eventType string, handler EventHandler) string
}
