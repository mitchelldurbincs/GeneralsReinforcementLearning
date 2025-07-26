package events

import (
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
)

// Event type constants
const (
	TypeGameStarted      = "game.started"
	TypeGameEnded        = "game.ended"
	TypeTurnStarted      = "turn.started"
	TypeTurnEnded        = "turn.ended"
	TypeActionSubmitted  = "action.submitted"
	TypeActionProcessed  = "action.processed"
	TypeActionRejected   = "action.rejected"
	TypeMoveSubmitted    = "move.submitted"
	TypeMoveValidated    = "move.validated"
	TypeMoveExecuted     = "move.executed"
	TypeCombatStarted    = "combat.started"
	TypeCombatResolved   = "combat.resolved"
	TypeTilesCaptured    = "tiles.captured"
	TypePlayerJoined     = "player.joined"
	TypePlayerEliminated = "player.eliminated"
	TypePlayerWon        = "player.won"
	TypeProductionApplied = "production.applied"
	TypeCityProduced     = "city.produced"
	TypeGeneralProduced  = "general.produced"
	TypeStateTransition  = "state.transition"
)

// GameStartedEvent is published when a new game begins
type GameStartedEvent struct {
	BaseEvent
	Metadata   EventMetadata
	NumPlayers int
	MapWidth   int
	MapHeight  int
}

// NewGameStartedEvent creates a new GameStartedEvent
func NewGameStartedEvent(gameID string, numPlayers, width, height int) *GameStartedEvent {
	return &GameStartedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeGameStarted,
			Time:      time.Now(),
			Game:      gameID,
		},
		NumPlayers: numPlayers,
		MapWidth:   width,
		MapHeight:  height,
	}
}

// GameEndedEvent is published when a game ends
type GameEndedEvent struct {
	BaseEvent
	Metadata EventMetadata
	Winner   int
	Duration time.Duration
	FinalTurn int
}

// NewGameEndedEvent creates a new GameEndedEvent
func NewGameEndedEvent(gameID string, winner int, duration time.Duration, finalTurn int) *GameEndedEvent {
	return &GameEndedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeGameEnded,
			Time:      time.Now(),
			Game:      gameID,
		},
		Winner:    winner,
		Duration:  duration,
		FinalTurn: finalTurn,
	}
}

// TurnStartedEvent is published at the beginning of each turn
type TurnStartedEvent struct {
	BaseEvent
	Metadata   EventMetadata
	TurnNumber int
}

// NewTurnStartedEvent creates a new TurnStartedEvent
func NewTurnStartedEvent(gameID string, turn int) *TurnStartedEvent {
	return &TurnStartedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeTurnStarted,
			Time:      time.Now(),
			Game:      gameID,
		},
		Metadata: EventMetadata{
			Turn: turn,
		},
		TurnNumber: turn,
	}
}

// TurnEndedEvent is published at the end of each turn
type TurnEndedEvent struct {
	BaseEvent
	Metadata      EventMetadata
	TurnNumber    int
	ActionsCount  int
	ProcessedTime time.Duration
}

// NewTurnEndedEvent creates a new TurnEndedEvent
func NewTurnEndedEvent(gameID string, turn int, actionsCount int, processedTime time.Duration) *TurnEndedEvent {
	return &TurnEndedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeTurnEnded,
			Time:      time.Now(),
			Game:      gameID,
		},
		Metadata: EventMetadata{
			Turn: turn,
		},
		TurnNumber:    turn,
		ActionsCount:  actionsCount,
		ProcessedTime: processedTime,
	}
}

// ActionSubmittedEvent is published when a player submits an action
type ActionSubmittedEvent struct {
	BaseEvent
	Metadata EventMetadata
	PlayerID int
	Action   core.Action
}

// NewActionSubmittedEvent creates a new ActionSubmittedEvent
func NewActionSubmittedEvent(gameID string, playerID int, action core.Action, turn int) *ActionSubmittedEvent {
	return &ActionSubmittedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeActionSubmitted,
			Time:      time.Now(),
			Game:      gameID,
		},
		Metadata: EventMetadata{
			PlayerID: playerID,
			Turn:     turn,
		},
		PlayerID: playerID,
		Action:   action,
	}
}

// ActionProcessedEvent is published after an action is successfully processed
type ActionProcessedEvent struct {
	BaseEvent
	Metadata EventMetadata
	PlayerID int
	Action   core.Action
	Result   string
}

// NewActionProcessedEvent creates a new ActionProcessedEvent
func NewActionProcessedEvent(gameID string, playerID int, action core.Action, result string, turn int) *ActionProcessedEvent {
	return &ActionProcessedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeActionProcessed,
			Time:      time.Now(),
			Game:      gameID,
		},
		Metadata: EventMetadata{
			PlayerID: playerID,
			Turn:     turn,
		},
		PlayerID: playerID,
		Action:   action,
		Result:   result,
	}
}

// MoveExecutedEvent is published when a move is executed
type MoveExecutedEvent struct {
	BaseEvent
	Metadata    EventMetadata
	PlayerID    int
	From        core.Coordinate
	To          core.Coordinate
	ArmiesMoved int
	Half        bool
}

// GetFromX returns the X coordinate of the From position (backward compatibility)
func (e *MoveExecutedEvent) GetFromX() int {
	return e.From.X
}

// GetFromY returns the Y coordinate of the From position (backward compatibility)
func (e *MoveExecutedEvent) GetFromY() int {
	return e.From.Y
}

// GetToX returns the X coordinate of the To position (backward compatibility)
func (e *MoveExecutedEvent) GetToX() int {
	return e.To.X
}

// GetToY returns the Y coordinate of the To position (backward compatibility)
func (e *MoveExecutedEvent) GetToY() int {
	return e.To.Y
}

// NewMoveExecutedEvent creates a new MoveExecutedEvent
func NewMoveExecutedEvent(gameID string, playerID int, fromX, fromY, toX, toY int, armies int, half bool, turn int) *MoveExecutedEvent {
	return &MoveExecutedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeMoveExecuted,
			Time:      time.Now(),
			Game:      gameID,
		},
		Metadata: EventMetadata{
			PlayerID: playerID,
			Turn:     turn,
		},
		PlayerID:    playerID,
		From:        core.NewCoordinate(fromX, fromY),
		To:          core.NewCoordinate(toX, toY),
		ArmiesMoved: armies,
		Half:        half,
	}
}

// CombatResolvedEvent is published when combat occurs
type CombatResolvedEvent struct {
	BaseEvent
	Metadata        EventMetadata
	AttackerID      int
	DefenderID      int
	Location        core.Coordinate
	AttackerArmies  int
	DefenderArmies  int
	AttackerLosses  int
	DefenderLosses  int
	TileCaptured    bool
}

// GetLocationX returns the X coordinate of the combat location (backward compatibility)
func (e *CombatResolvedEvent) GetLocationX() int {
	return e.Location.X
}

// GetLocationY returns the Y coordinate of the combat location (backward compatibility)
func (e *CombatResolvedEvent) GetLocationY() int {
	return e.Location.Y
}

// NewCombatResolvedEvent creates a new CombatResolvedEvent
func NewCombatResolvedEvent(gameID string, attacker, defender int, locationX, locationY int, 
	attackerArmies, defenderArmies, attackerLosses, defenderLosses int, captured bool, turn int) *CombatResolvedEvent {
	return &CombatResolvedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeCombatResolved,
			Time:      time.Now(),
			Game:      gameID,
		},
		Metadata: EventMetadata{
			PlayerID: attacker,
			Turn:     turn,
		},
		AttackerID:      attacker,
		DefenderID:      defender,
		Location:        core.NewCoordinate(locationX, locationY),
		AttackerArmies:  attackerArmies,
		DefenderArmies:  defenderArmies,
		AttackerLosses:  attackerLosses,
		DefenderLosses:  defenderLosses,
		TileCaptured:    captured,
	}
}

// PlayerEliminatedEvent is published when a player is eliminated
type PlayerEliminatedEvent struct {
	BaseEvent
	Metadata     EventMetadata
	PlayerID     int
	EliminatedBy int
	FinalRank    int
}

// NewPlayerEliminatedEvent creates a new PlayerEliminatedEvent
func NewPlayerEliminatedEvent(gameID string, playerID, eliminatedBy int, rank int, turn int) *PlayerEliminatedEvent {
	return &PlayerEliminatedEvent{
		BaseEvent: BaseEvent{
			EventType: TypePlayerEliminated,
			Time:      time.Now(),
			Game:      gameID,
		},
		Metadata: EventMetadata{
			PlayerID: playerID,
			Turn:     turn,
		},
		PlayerID:     playerID,
		EliminatedBy: eliminatedBy,
		FinalRank:    rank,
	}
}

// ProductionAppliedEvent is published when production is applied
type ProductionAppliedEvent struct {
	BaseEvent
	Metadata         EventMetadata
	TilesProduced    int
	CitiesProduced   int
	GeneralsProduced int
}

// NewProductionAppliedEvent creates a new ProductionAppliedEvent
func NewProductionAppliedEvent(gameID string, tiles, cities, generals int, turn int) *ProductionAppliedEvent {
	return &ProductionAppliedEvent{
		BaseEvent: BaseEvent{
			EventType: TypeProductionApplied,
			Time:      time.Now(),
			Game:      gameID,
		},
		Metadata: EventMetadata{
			Turn: turn,
		},
		TilesProduced:    tiles,
		CitiesProduced:   cities,
		GeneralsProduced: generals,
	}
}

// StateTransitionEvent is published when the game state machine transitions between phases
type StateTransitionEvent struct {
	BaseEvent
	FromPhase string
	ToPhase   string
	Reason    string
}

// NewStateTransitionEvent creates a new StateTransitionEvent
func NewStateTransitionEvent(gameID, fromPhase, toPhase, reason string) *StateTransitionEvent {
	return &StateTransitionEvent{
		BaseEvent: BaseEvent{
			EventType: TypeStateTransition,
			Time:      time.Now(),
			Game:      gameID,
		},
		FromPhase: fromPhase,
		ToPhase:   toPhase,
		Reason:    reason,
	}
}