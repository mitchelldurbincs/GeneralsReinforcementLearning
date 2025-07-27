"""
Shared type definitions and dataclasses for the Generals.io agent framework.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

# Import the generated protobuf types
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))
from generals_pb.game.v1 import game_pb2


class AgentState(Enum):
    """State of an agent during its lifecycle."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    WAITING_FOR_GAME = "waiting_for_game"
    WAITING_FOR_PLAYERS = "waiting_for_players"
    IN_GAME = "in_game"
    GAME_ENDED = "game_ended"
    ERROR = "error"


class GameStatus(Enum):
    """Status of a game."""
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class AgentContext:
    """
    Context information for an agent's current state.
    Encapsulates all state needed to track an agent's progress.
    """
    state: AgentState = AgentState.DISCONNECTED
    game_id: Optional[str] = None
    player_id: Optional[str] = None
    player_token: Optional[str] = None
    last_turn: int = -1
    last_game_status: Optional[GameStatus] = None
    current_game_state: Optional[game_pb2.GameState] = None
    error: Optional[Exception] = None
    
    def reset_game_context(self):
        """Reset game-specific context while maintaining connection state."""
        self.game_id = None
        self.player_id = None
        self.player_token = None
        self.last_turn = -1
        self.last_game_status = None
        self.current_game_state = None
        if self.state != AgentState.DISCONNECTED:
            self.state = AgentState.CONNECTED


@dataclass
class Position:
    """2D position on the game board."""
    x: int
    y: int
    
    def to_proto(self):
        """Convert to protobuf Coordinate."""
        # Import here to avoid circular imports
        from generals_pb.common.v1 import common_pb2
        return common_pb2.Coordinate(x=self.x, y=self.y)
    
    @classmethod
    def from_proto(cls, proto) -> 'Position':
        """Create from protobuf Coordinate."""
        return cls(x=proto.x, y=proto.y)
    
    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class Move:
    """Represents a move action in the game."""
    from_pos: Position
    to_pos: Position
    split: bool = False
    
    def to_proto(self) -> game_pb2.Action:
        """Convert to protobuf Action."""
        from generals_pb.common.v1 import common_pb2
        action = game_pb2.Action(
            type=common_pb2.ACTION_TYPE_MOVE,
            to=self.to_pos.to_proto(),
            half=self.split,
            turn_number=0  # Will be set by the game
        )
        # Use CopyFrom to set the 'from' field (reserved keyword)
        getattr(action, 'from').CopyFrom(self.from_pos.to_proto())
        return action


@dataclass
class GameStats:
    """Statistics about a completed game."""
    game_id: str
    winner_id: Optional[str]
    total_turns: int
    player_stats: Dict[str, 'PlayerStats']
    duration_seconds: float
    
    
@dataclass
class PlayerStats:
    """Statistics for a single player in a game."""
    player_id: str
    player_name: str
    final_army_count: int
    final_tile_count: int
    peak_army_count: int
    peak_tile_count: int
    eliminated_turn: Optional[int] = None
    finish_position: Optional[int] = None


@dataclass
class TileInfo:
    """Information about a tile from the agent's perspective."""
    position: Position
    tile_type: str  # "empty", "mountain", "city", "general"
    owner_id: Optional[str] = None
    army_size: int = 0
    is_visible: bool = True
    last_seen_turn: Optional[int] = None
    
    @classmethod
    def from_proto(cls, proto: game_pb2.Tile, position: Position) -> 'TileInfo':
        """Create from protobuf Tile."""
        # Import common_pb2 for TileType enum
        from generals_pb.common.v1 import common_pb2
        tile_type = common_pb2.TileType.Name(proto.type).lower().replace('tile_type_', '')
        return cls(
            position=position,
            tile_type=tile_type,
            owner_id=proto.owner_id if proto.owner_id else None,
            army_size=proto.army_count,
            is_visible=True  # Tiles in state are always visible
        )


@dataclass 
class BoardView:
    """
    A simplified view of the game board from an agent's perspective.
    Provides easier access to game state information.
    """
    width: int
    height: int
    tiles: Dict[Position, TileInfo]
    player_id: str
    turn: int
    
    @classmethod
    def from_game_state(cls, state: game_pb2.GameState, player_id: str) -> 'BoardView':
        """Create a BoardView from protobuf GameState."""
        tiles = {}
        for y in range(state.board.height):
            for x in range(state.board.width):
                idx = y * state.board.width + x
                if idx < len(state.board.tiles):
                    pos = Position(x, y)
                    tiles[pos] = TileInfo.from_proto(state.board.tiles[idx], pos)
        
        return cls(
            width=state.board.width,
            height=state.board.height,
            tiles=tiles,
            player_id=player_id,
            turn=state.turn
        )
    
    def get_tile(self, x: int, y: int) -> Optional[TileInfo]:
        """Get tile info at position."""
        return self.tiles.get(Position(x, y))
    
    def get_owned_tiles(self) -> List[TileInfo]:
        """Get all tiles owned by this player."""
        return [tile for tile in self.tiles.values() if tile.owner_id == self.player_id]
    
    def get_enemy_tiles(self) -> List[TileInfo]:
        """Get all tiles owned by enemies."""
        return [tile for tile in self.tiles.values() 
                if tile.owner_id and tile.owner_id != self.player_id]
    
    def get_neutral_tiles(self) -> List[TileInfo]:
        """Get all neutral tiles (cities without owners)."""
        return [tile for tile in self.tiles.values() 
                if not tile.owner_id and tile.tile_type in ["city", "empty"]]


# Type aliases for clarity
GameId = str
PlayerId = str 
PlayerToken = str
TurnNumber = int