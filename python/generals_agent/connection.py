"""
GameConnection class for managing gRPC connections to the game server.
Handles connection lifecycle, retries, and error handling.
"""
import grpc
import logging
import time
from typing import Optional, Callable, TypeVar, Any
from functools import wraps

# Import the generated gRPC stubs
from generals_pb.game.v1 import game_pb2_grpc

T = TypeVar('T')

class GameConnection:
    """Handles gRPC connection lifecycle to the game server."""
    
    def __init__(self, server_address: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize a new game connection.
        
        Args:
            server_address: The gRPC server address (e.g., "localhost:50051")
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
        """
        self.server_address = server_address
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[game_pb2_grpc.GameServiceStub] = None
    
    def connect(self) -> game_pb2_grpc.GameServiceStub:
        """
        Establish a connection to the game server.
        
        Returns:
            The gRPC stub for making RPC calls
            
        Raises:
            ConnectionError: If unable to connect after max retries
        """
        if self._stub is not None:
            return self._stub
            
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting to connect to {self.server_address} (attempt {attempt + 1}/{self.max_retries})")
                self._channel = grpc.insecure_channel(self.server_address)
                self._stub = game_pb2_grpc.GameServiceStub(self._channel)
                
                # Test the connection with a simple call
                # We'll need to add a health check or use an existing lightweight call
                self.logger.info(f"Successfully connected to {self.server_address}")
                return self._stub
                
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise ConnectionError(f"Failed to connect to {self.server_address} after {self.max_retries} attempts")
    
    def disconnect(self):
        """Close the gRPC connection."""
        if self._channel:
            try:
                self._channel.close()
                self.logger.info(f"Disconnected from {self.server_address}")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
            finally:
                self._channel = None
                self._stub = None
    
    def with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with automatic retry on connection failures.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The return value of the function
            
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except grpc.RpcError as e:
                last_exception = e
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    self.logger.warning(f"RPC failed due to unavailable server (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        # Try to reconnect
                        self._stub = None
                        self.connect()
                        time.sleep(self.retry_delay)
                else:
                    # Don't retry for other error types
                    raise
            except Exception as e:
                last_exception = e
                self.logger.error(f"Unexpected error in RPC call: {e}")
                raise
        
        if last_exception:
            raise last_exception
        
    def ensure_connected(self) -> game_pb2_grpc.GameServiceStub:
        """
        Ensure the connection is established, connecting if necessary.
        
        Returns:
            The gRPC stub for making RPC calls
        """
        if self._stub is None:
            return self.connect()
        return self._stub
    
    @property
    def is_connected(self) -> bool:
        """Check if the connection is currently established."""
        return self._stub is not None
    
    def __enter__(self):
        """Context manager entry - establish connection."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.disconnect()
        return False


def with_connection_retry(method: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for methods that need automatic connection retry.
    
    This decorator assumes the method is part of a class that has a
    'connection' attribute of type GameConnection.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs) -> T:
        if not hasattr(self, 'connection'):
            raise AttributeError(f"{self.__class__.__name__} must have a 'connection' attribute")
        return self.connection.with_retry(method, self, *args, **kwargs)
    return wrapper