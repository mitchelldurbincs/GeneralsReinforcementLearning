package gameserver

import (
	"fmt"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/core"
	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game/states"
	commonv1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/common/v1"
	gamev1 "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/game/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// convertTileType converts from core tile type (int) to protobuf TileType
func convertTileType(t int) commonv1.TileType {
	switch t {
	case core.TileNormal:
		return commonv1.TileType_TILE_TYPE_NORMAL
	case core.TileGeneral:
		return commonv1.TileType_TILE_TYPE_GENERAL
	case core.TileCity:
		return commonv1.TileType_TILE_TYPE_CITY
	case core.TileMountain:
		return commonv1.TileType_TILE_TYPE_MOUNTAIN
	default:
		return commonv1.TileType_TILE_TYPE_UNSPECIFIED
	}
}

// convertPhaseToProto converts internal state machine phase to proto phase
func convertPhaseToProto(phase states.GamePhase) commonv1.GamePhase {
	switch phase {
	case states.PhaseInitializing:
		return commonv1.GamePhase_GAME_PHASE_INITIALIZING
	case states.PhaseLobby:
		return commonv1.GamePhase_GAME_PHASE_LOBBY
	case states.PhaseStarting:
		return commonv1.GamePhase_GAME_PHASE_STARTING
	case states.PhaseRunning:
		return commonv1.GamePhase_GAME_PHASE_RUNNING
	case states.PhasePaused:
		return commonv1.GamePhase_GAME_PHASE_PAUSED
	case states.PhaseEnding:
		return commonv1.GamePhase_GAME_PHASE_ENDING
	case states.PhaseEnded:
		return commonv1.GamePhase_GAME_PHASE_ENDED
	case states.PhaseError:
		return commonv1.GamePhase_GAME_PHASE_ERROR
	case states.PhaseReset:
		return commonv1.GamePhase_GAME_PHASE_RESET
	default:
		return commonv1.GamePhase_GAME_PHASE_UNSPECIFIED
	}
}

// convertProtoToPhase converts proto phase to internal state machine phase
// nolint:unused // Will be used when state machine is fully integrated with gRPC
func convertProtoToPhase(phase commonv1.GamePhase) states.GamePhase {
	switch phase {
	case commonv1.GamePhase_GAME_PHASE_INITIALIZING:
		return states.PhaseInitializing
	case commonv1.GamePhase_GAME_PHASE_LOBBY:
		return states.PhaseLobby
	case commonv1.GamePhase_GAME_PHASE_STARTING:
		return states.PhaseStarting
	case commonv1.GamePhase_GAME_PHASE_RUNNING:
		return states.PhaseRunning
	case commonv1.GamePhase_GAME_PHASE_PAUSED:
		return states.PhasePaused
	case commonv1.GamePhase_GAME_PHASE_ENDING:
		return states.PhaseEnding
	case commonv1.GamePhase_GAME_PHASE_ENDED:
		return states.PhaseEnded
	case commonv1.GamePhase_GAME_PHASE_ERROR:
		return states.PhaseError
	case commonv1.GamePhase_GAME_PHASE_RESET:
		return states.PhaseReset
	default:
		return states.PhaseInitializing
	}
}

// mapPhaseToStatus maps game phase to legacy game status for backward compatibility
func mapPhaseToStatus(phase commonv1.GamePhase) commonv1.GameStatus {
	switch phase {
	case commonv1.GamePhase_GAME_PHASE_LOBBY:
		return commonv1.GameStatus_GAME_STATUS_WAITING
	case commonv1.GamePhase_GAME_PHASE_RUNNING:
		return commonv1.GameStatus_GAME_STATUS_IN_PROGRESS
	case commonv1.GamePhase_GAME_PHASE_ENDED:
		return commonv1.GameStatus_GAME_STATUS_FINISHED
	case commonv1.GamePhase_GAME_PHASE_ERROR:
		return commonv1.GameStatus_GAME_STATUS_CANCELLED
	default:
		// For other phases, map to waiting or in progress based on whether game has started
		if phase == commonv1.GamePhase_GAME_PHASE_STARTING || phase == commonv1.GamePhase_GAME_PHASE_PAUSED ||
			phase == commonv1.GamePhase_GAME_PHASE_ENDING {
			return commonv1.GameStatus_GAME_STATUS_IN_PROGRESS
		}
		return commonv1.GameStatus_GAME_STATUS_WAITING
	}
}

// convertProtoAction converts a protobuf action to a core game action
func convertProtoAction(protoAction *gamev1.Action, playerID int32) (core.Action, error) {
	if protoAction == nil {
		return nil, nil // No action this turn
	}

	switch protoAction.Type {
	case commonv1.ActionType_ACTION_TYPE_MOVE:
		if protoAction.From == nil || protoAction.To == nil {
			return nil, status.Errorf(codes.InvalidArgument, "move action for player %d requires from and to coordinates", playerID)
		}

		return &core.MoveAction{
			PlayerID: int(playerID),
			FromX:    int(protoAction.From.X),
			FromY:    int(protoAction.From.Y),
			ToX:      int(protoAction.To.X),
			ToY:      int(protoAction.To.Y),
			MoveAll:  !protoAction.Half, // In proto, half=true means move half; in core, MoveAll=true means move all
		}, nil

	case commonv1.ActionType_ACTION_TYPE_UNSPECIFIED:
		// No action this turn (wait/skip)
		return nil, nil

	default:
		return nil, status.Errorf(codes.InvalidArgument, "unsupported action type %v for player %d", protoAction.Type, playerID)
	}
}

// generatePlayerColor generates a simple color for a player based on their index
func generatePlayerColor(playerIndex int) string {
	return fmt.Sprintf("#%06X", playerIndex*0x333333)
}
