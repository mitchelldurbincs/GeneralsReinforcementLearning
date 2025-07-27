#!/usr/bin/env python3
"""Test script to validate imports of the new architecture components."""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing imports of new architecture components...")

try:
    # Import modules directly to avoid __init__.py
    import generals_agent.types as types
    import generals_agent.errors as errors
    import generals_agent.polling as polling
    
    # Test types module (no external dependencies)
    AgentState = types.AgentState
    Position = types.Position
    print("✓ types.py imports successfully")
    
    # Test Position class
    pos = Position(5, 10)
    print(f"✓ Position created: {pos}")
    
    # Test errors module  
    GameError = errors.GameError
    print("✓ errors.py imports successfully")
    
    # Test polling module
    FixedIntervalPolling = polling.FixedIntervalPolling
    print("✓ polling.py imports successfully")
    
    # Test a polling strategy
    strategy = FixedIntervalPolling(0.1)
    delay = strategy.get_next_delay(0)
    print(f"✓ FixedIntervalPolling works: delay={delay}")
    
    print("\nAll basic imports successful!")
    
except Exception as e:
    print(f"\n✗ Import failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()