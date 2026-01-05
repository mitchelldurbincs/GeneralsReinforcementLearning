# Human Player Guide

This guide explains how to play the Generals.io-style game with human controls.

## Running the Game

### Basic Usage (Human vs AI)
```bash
go run cmd/ui_client/main.go
```
This starts a game with Player 0 (Red) as human-controlled and Player 1 (Blue) as AI.

### Command Line Options
```bash
# Specify which player is human (0-based index)
go run cmd/ui_client/main.go -human=1

# Set number of players (2-4)
go run cmd/ui_client/main.go -players=4

# Set map size
go run cmd/ui_client/main.go -width=25 -height=20

# Run original AI-only mode
go run cmd/ui_client/main.go -ai-only
```

## Controls

### Mouse Controls
- **Left Click**: 
  - First click: Select a tile with your armies
  - Second click: Move to an adjacent tile
  - Click on selected tile to deselect
- **Right Click**: Cancel current selection

### Keyboard Controls
- **WASD / Arrow Keys**: Move from the selected tile
- **Shift** (hold): Move half the army (release for full move)
- **ESC**: Deselect current tile

## Gameplay

### How to Play
1. The game advances automatically; you can queue moves at any time
2. Click one of your tiles that has armies (number > 1)
3. The selected tile is highlighted in yellow
4. Adjacent valid moves are highlighted in green
5. Click an adjacent tile (or use WASD/Arrow keys) to move
6. The selection follows the destination, so you can chain moves
7. Right click or press ESC to cancel selection

### Visual Indicators
- **Yellow Border**: Currently selected tile
- **Green Overlay**: Valid move destinations
- **Red Overlay**: Invalid moves (mountains)
- **White Overlay**: Hover effect on interactive tiles
- **Dark Overlay**: Fog of war (enemy territory you can't see)

### Movement Rules
- You can only move armies from tiles you own
- Must have at least 2 armies to move (1 stays behind)
- Can only move to orthogonally adjacent tiles (up, down, left, right)
- Cannot move to mountains (dark gray tiles)
- Moving to enemy tiles captures them if you have more armies

### Game Objectives
- Capture enemy generals (white squares) to eliminate players
- Control cities (gray squares) for extra army production
- Build the largest army and control the most territory

## Tips
- Cities produce 1 army per turn when captured
- Your general also produces 1 army per turn
- Normal tiles produce 1 army every 25 turns
- Use half-army moves to expand territory while keeping defenses
- Fog of war hides enemy movements - scout carefully!
