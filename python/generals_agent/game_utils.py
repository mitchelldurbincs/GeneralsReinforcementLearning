"""Game utility functions for move validation and game state analysis"""

from typing import List, Tuple, Optional, Set

from generals_pb.game.v1 import game_pb2
from generals_pb.common.v1 import common_pb2


def get_owned_tiles(game_state: game_pb2.GameState, player_id: str) -> List[Tuple[int, int]]:
    """Get all tiles owned by a player
    
    Args:
        game_state: Current game state
        player_id: Player ID to check ownership for
        
    Returns:
        List of (x, y) coordinates of owned tiles
    """
    owned_tiles = []
    player_idx = int(player_id)
    
    for y in range(game_state.board.height):
        for x in range(game_state.board.width):
            idx = y * game_state.board.width + x
            tile = game_state.board.tiles[idx]
            if tile.owner_id == player_idx and tile.type != common_pb2.TILE_TYPE_MOUNTAIN and tile.owner_id != -1:
                owned_tiles.append((x, y))
                
    return owned_tiles


def get_tile_at(game_state: game_pb2.GameState, x: int, y: int) -> Optional[game_pb2.Tile]:
    """Get tile at specific coordinates
    
    Args:
        game_state: Current game state
        x: X coordinate
        y: Y coordinate
        
    Returns:
        Tile at position or None if out of bounds
    """
    if x < 0 or x >= game_state.board.width or y < 0 or y >= game_state.board.height:
        return None
        
    idx = y * game_state.board.width + x
    return game_state.board.tiles[idx]


def get_adjacent_positions(x: int, y: int) -> List[Tuple[int, int]]:
    """Get adjacent positions (up, down, left, right)
    
    Args:
        x: X coordinate
        y: Y coordinate
        
    Returns:
        List of adjacent (x, y) positions
    """
    return [(x, y-1), (x+1, y), (x, y+1), (x-1, y)]


def is_valid_move(game_state: game_pb2.GameState, action: game_pb2.Action, 
                  player_id: str) -> bool:
    """Check if a move action is valid
    
    Args:
        game_state: Current game state
        action: Action to validate
        player_id: Player attempting the action
        
    Returns:
        True if move is valid, False otherwise
    """
    if action.type not in [common_pb2.ACTION_TYPE_MOVE]:
        return False
        
    # Check source tile
    from_coord = getattr(action, 'from')
    source_tile = get_tile_at(game_state, from_coord.x, from_coord.y)
    if not source_tile:
        return False
        
    # Must own the source tile
    if source_tile.owner_id != int(player_id):
        return False
        
    # Must have enough armies
    min_armies = 3 if action.half else 2
    if source_tile.army_count < min_armies:
        return False
        
    # Check destination is adjacent
    from_coord = getattr(action, 'from')
    if abs(from_coord.x - action.to.x) + abs(from_coord.y - action.to.y) != 1:
        return False
        
    # Check destination tile
    dest_tile = get_tile_at(game_state, action.to.x, action.to.y)
    if not dest_tile:
        return False
        
    # Can't move to mountains
    if dest_tile.type == common_pb2.TILE_TYPE_MOUNTAIN:
        return False
        
    return True


def get_valid_moves(game_state: game_pb2.GameState, player_id: str) -> List[game_pb2.Action]:
    """Get all valid moves for a player
    
    Args:
        game_state: Current game state
        player_id: Player to get moves for
        
    Returns:
        List of valid actions
    """
    valid_moves = []
    owned_tiles = get_owned_tiles(game_state, player_id)
    
    for x, y in owned_tiles:
        tile = get_tile_at(game_state, x, y)
        if not tile:
            continue
            
        # Skip if not enough armies
        if tile.army_count < 2:
            continue
            
        # Check all adjacent positions
        for to_x, to_y in get_adjacent_positions(x, y):
            # Try regular move
            action = game_pb2.Action(
                type=common_pb2.ACTION_TYPE_MOVE,
                to=common_pb2.Coordinate(x=to_x, y=to_y),
                half=False,
                turn_number=game_state.turn
            )
            # Use CopyFrom to set the 'from' field
            getattr(action, 'from').CopyFrom(common_pb2.Coordinate(x=x, y=y))
            if is_valid_move(game_state, action, player_id):
                valid_moves.append(action)
                
            # Try half move if we have enough armies
            if tile.army_count >= 3:
                action_half = game_pb2.Action(
                    type=common_pb2.ACTION_TYPE_MOVE,
                    to=common_pb2.Coordinate(x=to_x, y=to_y),
                    half=True,
                    turn_number=game_state.turn
                )
                # Use CopyFrom to set the 'from' field
                getattr(action_half, 'from').CopyFrom(common_pb2.Coordinate(x=x, y=y))
                if is_valid_move(game_state, action_half, player_id):
                    valid_moves.append(action_half)
                    
    return valid_moves


def find_general_position(game_state: game_pb2.GameState, player_id: str) -> Optional[Tuple[int, int]]:
    """Find the position of a player's general
    
    Args:
        game_state: Current game state
        player_id: Player to find general for
        
    Returns:
        (x, y) position of general or None if not found
    """
    player_idx = int(player_id)
    
    for y in range(game_state.board.height):
        for x in range(game_state.board.width):
            idx = y * game_state.board.width + x
            tile = game_state.board.tiles[idx]
            if tile.type == common_pb2.TILE_TYPE_GENERAL and tile.owner_id == player_idx:
                return (x, y)
                
    return None


def get_visible_tiles(game_state: game_pb2.GameState, player_id: str) -> Set[Tuple[int, int]]:
    """Get all visible tiles for a player (if fog of war is enabled)
    
    Args:
        game_state: Current game state  
        player_id: Player to check visibility for
        
    Returns:
        Set of (x, y) coordinates of visible tiles
    """
    visible = set()
    player_idx = int(player_id)
    
    # If no fog of war, all tiles are visible
    # Check if the first tile has visibility tracking by checking if visible field exists and is False
    if not hasattr(game_state.board.tiles[0], 'visible') or not game_state.fog_of_war_enabled:
        for y in range(game_state.board.height):
            for x in range(game_state.board.width):
                visible.add((x, y))
        return visible
        
    # Otherwise check visibility flag on each tile
    for y in range(game_state.board.height):
        for x in range(game_state.board.width):
            idx = y * game_state.board.width + x
            tile = game_state.board.tiles[idx]
            if tile.visible:
                visible.add((x, y))
                
    return visible


def count_player_stats(game_state: game_pb2.GameState, player_id: str) -> dict:
    """Count various statistics for a player
    
    Args:
        game_state: Current game state
        player_id: Player to count stats for
        
    Returns:
        Dictionary with stats (tiles, armies, cities, etc.)
    """
    stats = {
        'tiles': 0,
        'armies': 0,
        'cities': 0,
        'has_general': False
    }
    
    player_idx = int(player_id)
    
    for tile in game_state.board.tiles:
        if tile.owner_id == player_idx:
            stats['tiles'] += 1
            stats['armies'] += tile.army_count
            
            if tile.type == common_pb2.TILE_TYPE_CITY:
                stats['cities'] += 1
            elif tile.type == common_pb2.TILE_TYPE_GENERAL:
                stats['has_general'] = True
                
    return stats