"""
Custom exception hierarchy for the Generals.io agent framework.
Provides specific error types for different failure scenarios.
"""
import grpc
from typing import Optional


class GameError(Exception):
    """Base exception for all game-related errors."""
    pass


class ConnectionError(GameError):
    """Network connection errors."""
    pass


class GameStateError(GameError):
    """Invalid game state errors."""
    pass


class ActionError(GameError):
    """Invalid action errors."""
    pass


class AuthenticationError(GameError):
    """Authentication and authorization errors."""
    pass


class GameNotFoundError(GameError):
    """Game does not exist or is no longer available."""
    pass


class GameFullError(GameError):
    """Game has reached maximum player capacity."""
    pass


class TurnTimeoutError(GameError):
    """Player took too long to submit an action."""
    pass


class InvalidMoveError(ActionError):
    """The submitted move violates game rules."""
    pass


def handle_grpc_errors(func):
    """
    Decorator for consistent gRPC error handling.
    Converts gRPC errors to our custom exception hierarchy.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except grpc.RpcError as e:
            # Map gRPC status codes to our custom exceptions
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise ConnectionError(f"Server unavailable: {e.details()}")
            elif e.code() == grpc.StatusCode.NOT_FOUND:
                raise GameNotFoundError(f"Game not found: {e.details()}")
            elif e.code() == grpc.StatusCode.UNAUTHENTICATED:
                raise AuthenticationError(f"Authentication failed: {e.details()}")
            elif e.code() == grpc.StatusCode.PERMISSION_DENIED:
                raise AuthenticationError(f"Permission denied: {e.details()}")
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise ActionError(f"Invalid request: {e.details()}")
            elif e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise GameFullError(f"Resource exhausted: {e.details()}")
            elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TurnTimeoutError(f"Deadline exceeded: {e.details()}")
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise GameStateError(f"Failed precondition: {e.details()}")
            else:
                # For any other gRPC errors, wrap in generic GameError
                raise GameError(f"RPC error: {e.code()} - {e.details()}")
    return wrapper


class ErrorContext:
    """
    Context manager for adding context to errors.
    
    Usage:
        with ErrorContext("joining game", game_id=game_id):
            # code that might raise exceptions
    """
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None and isinstance(exc_val, GameError):
            # Add context to the error message
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            new_message = f"Error while {self.operation}"
            if context_str:
                new_message += f" ({context_str})"
            new_message += f": {str(exc_val)}"
            
            # Create new exception of the same type with enhanced message
            exc_val.args = (new_message,)
        
        # Don't suppress the exception
        return False