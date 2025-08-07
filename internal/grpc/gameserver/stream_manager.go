package gameserver

import (
	"context"
	"sync"

	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/types/known/timestamppb"

	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
)

// streamClient represents a connected stream for a player
type streamClient struct {
	playerID   int32
	stream     gamev1.GameService_StreamGameServer
	ctx        context.Context
	cancelFunc context.CancelFunc
	updateChan chan *gamev1.GameUpdate
}

// StreamManager manages all stream clients for a game
type StreamManager struct {
	clients   map[int32]*streamClient // playerID -> stream client
	clientsMu sync.RWMutex            // Protects clients map
}

// NewStreamManager creates a new stream manager
func NewStreamManager() *StreamManager {
	return &StreamManager{
		clients: make(map[int32]*streamClient),
	}
}

// RegisterClient adds a new stream client for a player
func (sm *StreamManager) RegisterClient(client *streamClient) {
	sm.clientsMu.Lock()
	defer sm.clientsMu.Unlock()

	// Close any existing stream for this player
	if existing, exists := sm.clients[client.playerID]; exists {
		existing.cancelFunc()
		close(existing.updateChan)
	}

	sm.clients[client.playerID] = client

	log.Debug().
		Int32("player_id", client.playerID).
		Int("total_streams", len(sm.clients)).
		Msg("Stream client registered")
}

// UnregisterClient removes a stream client for a player
func (sm *StreamManager) UnregisterClient(playerID int32) {
	sm.clientsMu.Lock()
	defer sm.clientsMu.Unlock()

	if client, exists := sm.clients[playerID]; exists {
		client.cancelFunc()
		close(client.updateChan)
		delete(sm.clients, playerID)

		log.Debug().
			Int32("player_id", playerID).
			Int("remaining_streams", len(sm.clients)).
			Msg("Stream client unregistered")
	}
}

// SendToClient sends an update to a specific client
func (sm *StreamManager) SendToClient(playerID int32, update *gamev1.GameUpdate) {
	sm.clientsMu.RLock()
	client, exists := sm.clients[playerID]
	sm.clientsMu.RUnlock()

	if !exists {
		return
	}

	// Non-blocking send to avoid blocking the game
	select {
	case client.updateChan <- update:
		// Successfully queued update
	default:
		// Channel full, log warning
		log.Warn().
			Int32("player_id", playerID).
			Msg("Stream update channel full, dropping update")
	}
}

// BroadcastToAll sends an update to all connected clients
func (sm *StreamManager) BroadcastToAll(update *gamev1.GameUpdate) {
	sm.clientsMu.RLock()
	defer sm.clientsMu.RUnlock()

	for playerID, client := range sm.clients {
		select {
		case client.updateChan <- update:
			// Successfully queued event
		default:
			log.Warn().
				Int32("player_id", playerID).
				Msg("Stream event channel full, dropping event")
		}
	}
}

// GetClientCount returns the number of connected stream clients
func (sm *StreamManager) GetClientCount() int {
	sm.clientsMu.RLock()
	defer sm.clientsMu.RUnlock()
	return len(sm.clients)
}

// ForEachClient executes a function for each connected client
func (sm *StreamManager) ForEachClient(fn func(playerID int32, client *streamClient)) {
	sm.clientsMu.RLock()
	defer sm.clientsMu.RUnlock()

	for playerID, client := range sm.clients {
		fn(playerID, client)
	}
}

// CloseAll closes all stream clients
func (sm *StreamManager) CloseAll() {
	sm.clientsMu.Lock()
	defer sm.clientsMu.Unlock()

	for playerID, client := range sm.clients {
		client.cancelFunc()
		close(client.updateChan)
		delete(sm.clients, playerID)
	}
}

// BroadcastGameStarted sends a game started event to all connected clients
func (sm *StreamManager) BroadcastGameStarted() {
	if sm.GetClientCount() == 0 {
		return
	}

	event := &gamev1.GameEvent{
		Event: &gamev1.GameEvent_GameStarted{
			GameStarted: &gamev1.GameStartedEvent{
				StartedAt: timestamppb.Now(),
			},
		},
	}

	update := &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_Event{
			Event: event,
		},
		Timestamp: timestamppb.Now(),
	}

	sm.BroadcastToAll(update)
}

// BroadcastPlayerEliminated sends a player eliminated event to all connected clients
func (sm *StreamManager) BroadcastPlayerEliminated(playerID int32, eliminatedBy int32) {
	if sm.GetClientCount() == 0 {
		return
	}

	event := &gamev1.GameEvent{
		Event: &gamev1.GameEvent_PlayerEliminated{
			PlayerEliminated: &gamev1.PlayerEliminatedEvent{
				PlayerId:     playerID,
				EliminatedBy: eliminatedBy,
			},
		},
	}

	update := &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_Event{
			Event: event,
		},
		Timestamp: timestamppb.Now(),
	}

	sm.BroadcastToAll(update)
}

// BroadcastGameEnded sends a game ended event to all connected clients
func (sm *StreamManager) BroadcastGameEnded(winnerId int32) {
	if sm.GetClientCount() == 0 {
		return
	}

	event := &gamev1.GameEvent{
		Event: &gamev1.GameEvent_GameEnded{
			GameEnded: &gamev1.GameEndedEvent{
				WinnerId: winnerId,
				EndedAt:  timestamppb.Now(),
			},
		},
	}

	update := &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_Event{
			Event: event,
		},
		Timestamp: timestamppb.Now(),
	}

	sm.BroadcastToAll(update)
}

// BroadcastPhaseChanged broadcasts a phase change event to all connected clients
func (sm *StreamManager) BroadcastPhaseChanged(previousPhase, newPhase commonv1.GamePhase, reason string) {
	if sm.GetClientCount() == 0 {
		return
	}

	event := &gamev1.GameEvent{
		Event: &gamev1.GameEvent_PhaseChanged{
			PhaseChanged: &gamev1.PhaseChangedEvent{
				PreviousPhase: previousPhase,
				NewPhase:      newPhase,
				Reason:        reason,
			},
		},
	}

	update := &gamev1.GameUpdate{
		Update: &gamev1.GameUpdate_Event{
			Event: event,
		},
		Timestamp: timestamppb.Now(),
	}

	sm.BroadcastToAll(update)

	log.Info().
		Str("previous_phase", previousPhase.String()).
		Str("new_phase", newPhase.String()).
		Str("reason", reason).
		Msg("Phase changed event broadcast")
}
