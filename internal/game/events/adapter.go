package events

// EventPublisherAdapter adapts the EventBus to work with the processor.EventPublisher interface
type EventPublisherAdapter struct {
	bus *EventBus
}

// NewEventPublisherAdapter creates a new adapter
func NewEventPublisherAdapter(bus *EventBus) *EventPublisherAdapter {
	return &EventPublisherAdapter{bus: bus}
}

// Publish implements processor.EventPublisher
func (a *EventPublisherAdapter) Publish(event interface{}) {
	if e, ok := event.(Event); ok {
		a.bus.Publish(e)
	}
	// Silently ignore non-Event types
}