package events

import (
	"sync"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// EventBus is a synchronous event bus implementation
type EventBus struct {
	subscribers  map[string]Subscriber
	funcHandlers map[string][]EventHandler
	mu           sync.RWMutex
	logger       zerolog.Logger
}

// NewEventBus creates a new event bus instance
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers:  make(map[string]Subscriber),
		funcHandlers: make(map[string][]EventHandler),
		logger:       log.With().Str("component", "event_bus").Logger(),
	}
}

// Subscribe adds a new subscriber to the event bus
func (eb *EventBus) Subscribe(subscriber Subscriber) {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	eb.subscribers[subscriber.ID()] = subscriber
	eb.logger.Debug().
		Str("subscriber_id", subscriber.ID()).
		Msg("Subscriber added to event bus")
}

// Unsubscribe removes a subscriber from the event bus
func (eb *EventBus) Unsubscribe(subscriberID string) {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	delete(eb.subscribers, subscriberID)
	eb.logger.Debug().
		Str("subscriber_id", subscriberID).
		Msg("Subscriber removed from event bus")
}

// SubscribeFunc adds a function handler for specific event types
func (eb *EventBus) SubscribeFunc(eventType string, handler EventHandler) string {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	eb.funcHandlers[eventType] = append(eb.funcHandlers[eventType], handler)

	// Return a simple ID for the function handler
	handlerID := eventType + "_func_" + string(rune(len(eb.funcHandlers[eventType])))
	eb.logger.Debug().
		Str("event_type", eventType).
		Str("handler_id", handlerID).
		Msg("Function handler added to event bus")

	return handlerID
}

// Publish sends an event to all interested subscribers synchronously
func (eb *EventBus) Publish(event Event) {
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	eventType := event.Type()

	eb.logger.Debug().
		Str("event_type", eventType).
		Str("game_id", event.GameID()).
		Time("timestamp", event.Timestamp()).
		Msg("Publishing event")

	// Notify object subscribers
	for id, subscriber := range eb.subscribers {
		if subscriber.InterestedIn(eventType) {
			// Run synchronously but catch panics to prevent one subscriber from breaking others
			func() {
				defer func() {
					if r := recover(); r != nil {
						eb.logger.Error().
							Str("subscriber_id", id).
							Str("event_type", eventType).
							Interface("panic", r).
							Msg("Subscriber panicked while handling event")
					}
				}()
				subscriber.HandleEvent(event)
			}()
		}
	}

	// Notify function handlers
	if handlers, exists := eb.funcHandlers[eventType]; exists {
		for i, handler := range handlers {
			// Run synchronously but catch panics
			func() {
				defer func() {
					if r := recover(); r != nil {
						eb.logger.Error().
							Str("event_type", eventType).
							Int("handler_index", i).
							Interface("panic", r).
							Msg("Function handler panicked while handling event")
					}
				}()
				handler(event)
			}()
		}
	}
}

// PublishAsync publishes an event asynchronously (for future use)
// For now, it just calls Publish synchronously
func (eb *EventBus) PublishAsync(event Event) {
	// TODO: Implement async publishing in Phase 4
	eb.Publish(event)
}

// GetSubscriberCount returns the number of subscribers for debugging
func (eb *EventBus) GetSubscriberCount() int {
	eb.mu.RLock()
	defer eb.mu.RUnlock()
	return len(eb.subscribers)
}

// GetFuncHandlerCount returns the number of function handlers for a specific event type
func (eb *EventBus) GetFuncHandlerCount(eventType string) int {
	eb.mu.RLock()
	defer eb.mu.RUnlock()
	return len(eb.funcHandlers[eventType])
}
