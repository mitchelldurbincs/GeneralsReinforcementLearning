package experience

import (
	"math"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/game"
)

const (
	// Channel indices for tensor representation
	ChannelOwnArmies        = 0
	ChannelEnemyArmies      = 1
	ChannelOwnTerritory     = 2
	ChannelEnemyTerritory   = 3
	ChannelNeutralTerritory = 4
	ChannelCities           = 5
	ChannelMountains        = 6
	ChannelVisible          = 7
	ChannelFog              = 8
	NumChannels             = 9

	// Max army value for normalization
	MaxArmyValue = 1000.0
)

// Serializer converts game states to tensor representations
type Serializer struct {
	// Could add configuration options here
}

// NewSerializer creates a new state serializer
func NewSerializer() *Serializer {
	return &Serializer{}
}

// StateToTensor converts a game state to a multi-channel tensor from a player's perspective
func (s *Serializer) StateToTensor(state *game.GameState, playerID int) []float32 {
	width := state.Board.W
	height := state.Board.H
	tensorSize := NumChannels * width * height
	tensor := make([]float32, tensorSize)

	// Process each tile
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			tileIdx := y*width + x
			tile := &state.Board.T[tileIdx]

			// Check if tile is visible to player (if fog of war is disabled, all tiles are visible)
			isVisible := !state.FogOfWarEnabled || tile.IsVisibleTo(playerID)

			// Channel 7: Visible tiles
			if isVisible {
				tensor[s.getChannelIndex(ChannelVisible, x, y, width, height)] = 1.0
			}

			// Channel 8: Fog of war (not visible)
			if !isVisible {
				tensor[s.getChannelIndex(ChannelFog, x, y, width, height)] = 1.0
			}

			// Skip other channels if tile is not visible
			if !isVisible {
				continue
			}

			// Channel 6: Mountains
			if tile.IsMountain() {
				tensor[s.getChannelIndex(ChannelMountains, x, y, width, height)] = 1.0
				continue // Mountains can't have armies or ownership
			}

			// Channel 5: Cities
			if tile.IsCity() || tile.IsGeneral() {
				tensor[s.getChannelIndex(ChannelCities, x, y, width, height)] = 1.0
			}

			// Army and territory channels
			if tile.Owner == playerID {
				// Channel 0: Own armies (normalized)
				if tile.Army > 0 {
					normalizedArmies := float32(tile.Army) / MaxArmyValue
					if normalizedArmies > 1.0 {
						normalizedArmies = 1.0
					}
					tensor[s.getChannelIndex(ChannelOwnArmies, x, y, width, height)] = normalizedArmies
				}
				// Channel 2: Own territory
				tensor[s.getChannelIndex(ChannelOwnTerritory, x, y, width, height)] = 1.0
			} else if tile.Owner >= 0 {
				// Channel 1: Enemy armies (normalized)
				if tile.Army > 0 {
					normalizedArmies := float32(tile.Army) / MaxArmyValue
					if normalizedArmies > 1.0 {
						normalizedArmies = 1.0
					}
					tensor[s.getChannelIndex(ChannelEnemyArmies, x, y, width, height)] = normalizedArmies
				}
				// Channel 3: Enemy territory
				tensor[s.getChannelIndex(ChannelEnemyTerritory, x, y, width, height)] = 1.0
			} else {
				// Channel 4: Neutral territory
				tensor[s.getChannelIndex(ChannelNeutralTerritory, x, y, width, height)] = 1.0
			}
		}
	}

	return tensor
}

// GenerateActionMask creates a boolean mask of legal actions for a player
func (s *Serializer) GenerateActionMask(state *game.GameState, playerID int) []bool {
	width := state.Board.W
	height := state.Board.H

	// Actions are encoded as: (y * width + x) * 4 + direction
	// 4 directions per tile: Up, Down, Left, Right
	numActions := width * height * 4
	mask := make([]bool, numActions)

	// Check each tile
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			tileIdx := y*width + x
			tile := &state.Board.T[tileIdx]

			// Can only move from tiles we own with at least 2 armies
			if tile.Owner != playerID || tile.Army < 2 {
				continue
			}

			// Check each direction
			// Up
			if y > 0 {
				idx := s.actionToFlatIndex(x, y, 0, width)
				targetIdx := (y-1)*width + x
				targetTile := &state.Board.T[targetIdx]
				if !targetTile.IsMountain() {
					mask[idx] = true
				}
			}

			// Down
			if y < height-1 {
				idx := s.actionToFlatIndex(x, y, 1, width)
				targetIdx := (y+1)*width + x
				targetTile := &state.Board.T[targetIdx]
				if !targetTile.IsMountain() {
					mask[idx] = true
				}
			}

			// Left
			if x > 0 {
				idx := s.actionToFlatIndex(x, y, 2, width)
				targetIdx := y*width + (x - 1)
				targetTile := &state.Board.T[targetIdx]
				if !targetTile.IsMountain() {
					mask[idx] = true
				}
			}

			// Right
			if x < width-1 {
				idx := s.actionToFlatIndex(x, y, 3, width)
				targetIdx := y*width + (x + 1)
				targetTile := &state.Board.T[targetIdx]
				if !targetTile.IsMountain() {
					mask[idx] = true
				}
			}
		}
	}

	return mask
}

// ActionToIndex converts a game action to a flattened index
func (s *Serializer) ActionToIndex(action *game.Action, boardWidth int) int {
	// Calculate direction from From and To coordinates
	// Map direction to index: Up=0, Down=1, Left=2, Right=3
	dirIdx := 0

	dx := action.To.X - action.From.X
	dy := action.To.Y - action.From.Y

	if dy == -1 && dx == 0 {
		dirIdx = 0 // Up
	} else if dy == 1 && dx == 0 {
		dirIdx = 1 // Down
	} else if dy == 0 && dx == -1 {
		dirIdx = 2 // Left
	} else if dy == 0 && dx == 1 {
		dirIdx = 3 // Right
	}

	return s.actionToFlatIndex(action.From.X, action.From.Y, dirIdx, boardWidth)
}

// IndexToAction converts a flattened index back to coordinates and direction index
func (s *Serializer) IndexToAction(index int, boardWidth, boardHeight int) (fromX, fromY, toX, toY int) {
	// Reverse the encoding: index = (y * width + x) * 4 + direction
	dirIdx := index % 4
	tileIdx := index / 4

	fromX = tileIdx % boardWidth
	fromY = tileIdx / boardWidth

	// Calculate destination based on direction
	toX, toY = fromX, fromY
	switch dirIdx {
	case 0: // Up
		toY = fromY - 1
	case 1: // Down
		toY = fromY + 1
	case 2: // Left
		toX = fromX - 1
	case 3: // Right
		toX = fromX + 1
	}

	return fromX, fromY, toX, toY
}

// getChannelIndex calculates the index in the flattened tensor for a specific channel and position
func (s *Serializer) getChannelIndex(channel, x, y, width, height int) int {
	// Layout: [channel][height][width] in row-major order
	return channel*height*width + y*width + x
}

// actionToFlatIndex converts tile coordinates and direction to a flat action index
func (s *Serializer) actionToFlatIndex(x, y, direction, boardWidth int) int {
	return (y*boardWidth+x)*4 + direction
}

// NormalizeArmyValue normalizes army counts to [0, 1] range
func (s *Serializer) NormalizeArmyValue(armies int) float32 {
	normalized := float32(armies) / MaxArmyValue
	if normalized > 1.0 {
		return 1.0
	}
	return normalized
}

// ValidateAction checks if an action index is valid for the board dimensions
func (s *Serializer) ValidateAction(index int, boardWidth, boardHeight int) bool {
	maxIndex := boardWidth * boardHeight * 4
	return index >= 0 && index < maxIndex
}

// GetTensorShape returns the shape of the tensor representation
func (s *Serializer) GetTensorShape(boardWidth, boardHeight int) []int32 {
	return []int32{NumChannels, int32(boardHeight), int32(boardWidth)}
}

// ExtractFeatures extracts additional features that might be useful for the model
func (s *Serializer) ExtractFeatures(state *game.GameState, playerID int) map[string]float32 {
	features := make(map[string]float32)

	// Calculate relative army strength
	playerArmies := float32(0)
	enemyArmies := float32(0)

	for _, tile := range state.Board.T {
		if tile.Owner == playerID {
			playerArmies += float32(tile.Army)
		} else if tile.Owner >= 0 {
			enemyArmies += float32(tile.Army)
		}
	}

	// Avoid division by zero
	if enemyArmies > 0 {
		features["army_ratio"] = playerArmies / enemyArmies
	} else {
		features["army_ratio"] = float32(math.Inf(1))
	}

	features["turn_number"] = float32(state.Turn)
	features["total_armies"] = playerArmies

	return features
}
