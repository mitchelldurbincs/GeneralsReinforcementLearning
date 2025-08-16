#!/usr/bin/env python3
"""
Simple interactive client for the Generals game.
Shows basic game state and allows simple moves.
"""

import grpc
import time
from generals_pb.game.v1 import game_pb2, game_pb2_grpc
from generals_pb.common.v1 import common_pb2

def print_board(state, player_id):
    """Print the game board in a simple text format."""
    board = state.board
    width = board.width
    height = board.height
    
    print(f"\nTurn {state.turn} - Board {width}x{height}")
    print("=" * (width * 4 + 1))
    
    for y in range(height):
        row = "|"
        for x in range(width):
            idx = y * width + x
            tile = board.tiles[idx]
            
            # Determine tile display
            if not tile.visible:
                cell = " ? "  # Unknown/fog
            elif tile.type == common_pb2.TILE_TYPE_MOUNTAIN:
                cell = "###"  # Mountain
            elif tile.type == common_pb2.TILE_TYPE_GENERAL:
                if tile.owner_id == player_id:
                    cell = f"G{tile.army_count:2}"  # Our general
                elif tile.owner_id >= 0:
                    cell = f"g{tile.army_count:2}"  # Enemy general
                else:
                    cell = " . "  # Neutral
            elif tile.type == common_pb2.TILE_TYPE_CITY:
                if tile.owner_id == player_id:
                    cell = f"C{tile.army_count:2}"  # Our city
                elif tile.owner_id >= 0:
                    cell = f"c{tile.army_count:2}"  # Enemy city
                else:
                    cell = f"c{tile.army_count:2}"  # Neutral city
            else:  # Empty tile
                if tile.owner_id == player_id:
                    cell = f" {tile.army_count:2}"  # Our territory
                elif tile.owner_id >= 0:
                    cell = f"-{tile.army_count:2}"  # Enemy territory
                else:
                    cell = " . "  # Neutral
            
            row += cell + "|"
        print(row)
    
    print("=" * (width * 4 + 1))
    
    # Print player stats
    for player in state.players:
        status = "Active" if player.status == common_pb2.PLAYER_STATUS_ACTIVE else "Eliminated"
        print(f"Player {player.id} ({player.name}): {player.army_count} armies, {player.tile_count} tiles - {status}")

def run_simple_game():
    """Run a simple game demonstration."""
    channel = grpc.insecure_channel('localhost:50051')
    stub = game_pb2_grpc.GameServiceStub(channel)
    
    try:
        # Create a game
        print("Creating a new game...")
        create_response = stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(
                width=15,
                height=15,
                max_players=2,
                fog_of_war=True,
                turn_time_ms=500  # 0.5 second turns for demo
            )
        ))
        game_id = create_response.game_id
        print(f"Created game: {game_id}")
        
        # Join as player 1
        print("\nJoining as Player 1...")
        join1 = stub.JoinGame(game_pb2.JoinGameRequest(
            game_id=game_id,
            player_name="Human"
        ))
        player1_id = join1.player_id
        player1_token = join1.player_token
        
        # Join as player 2 (bot)
        print("Joining as Player 2 (Bot)...")
        join2 = stub.JoinGame(game_pb2.JoinGameRequest(
            game_id=game_id,
            player_name="Bot"
        ))
        player2_id = join2.player_id
        player2_token = join2.player_token
        
        # Game loop - watch for a few turns
        print("\nGame starting! Watching for 10 turns...")
        print("Legend: G=General, C=City, #=Mountain, ?=Fog")
        print("        Numbers show army count")
        print("        Your tiles show as positive numbers, enemy as negative")
        
        for turn in range(10):
            time.sleep(1)  # Wait a bit between checks
            
            # Get current game state
            state_response = stub.GetGameState(game_pb2.GetGameStateRequest(
                game_id=game_id,
                player_token=player1_token
            ))
            
            # Print the board
            print_board(state_response.state, player1_id)
            
            # Check if game ended
            if state_response.state.status != common_pb2.GAME_STATUS_IN_PROGRESS:
                print(f"\nGame ended! Status: {common_pb2.GameStatus.Name(state_response.state.status)}")
                if state_response.state.winner_id >= 0:
                    print(f"Winner: Player {state_response.state.winner_id}")
                break
        
        print("\nDemo complete!")
        
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()}: {e.details()}")
    finally:
        channel.close()

if __name__ == "__main__":
    print("Generals Game Simple Client")
    print("==========================")
    
    # Check if server is running
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        
        # Try a simple request to test connectivity
        stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(width=5, height=5)
        ))
        channel.close()
        
        # Server is running, proceed
        run_simple_game()
        
    except grpc.RpcError:
        print("\nError: Game server is not running!")
        print("Please start the server with: go run cmd/game_server/main.go")
        sys.exit(1)