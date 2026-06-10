#!/usr/bin/env python3
"""
Test the ParallelEnvPool: N envs in worker threads feeding a shared
thread-safe ReplayBuffer. Requires a running game server.
"""

import sys
import os
import random
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generals_gym import GeneralsEnv, ParallelEnvPool, ReplayBuffer


def test_replay_buffer_thread_safety():
    """Hammer the buffer from multiple threads; no server needed."""
    print("\nTesting ReplayBuffer thread safety...")

    capacity = 500
    pushes_per_thread = 1000
    n_threads = 4
    buffer = ReplayBuffer(capacity)
    errors = []

    def pusher(tid):
        try:
            for i in range(pushes_per_thread):
                buffer.push(np.zeros((9, 5, 5), dtype=np.float32), i, 0.5, np.zeros((9, 5, 5), dtype=np.float32), False)
                if len(buffer) >= 32:
                    batch = buffer.sample(32)
                    assert len(batch) == 32
        except Exception as e:
            errors.append((tid, e))

    threads = [threading.Thread(target=pusher, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert buffer.total_pushed == n_threads * pushes_per_thread, \
        f"Expected {n_threads * pushes_per_thread} pushes, got {buffer.total_pushed}"
    assert len(buffer) == capacity, f"Expected buffer at capacity {capacity}, got {len(buffer)}"

    print(f"  ✓ {buffer.total_pushed} pushes from {n_threads} threads, buffer at capacity {len(buffer)}")
    return True


def test_parallel_env_pool():
    """Run a small pool against the live server and check collected data."""
    num_envs = 2
    board_size = 5
    target_episodes = 4
    timeout_s = 90

    print(f"\nTesting ParallelEnvPool with {num_envs} envs on {board_size}x{board_size}...")

    buffer = ReplayBuffer(capacity=10000)

    def env_factory(worker_id):
        return GeneralsEnv(
            server_address="localhost:50051",
            board_width=board_size,
            board_height=board_size,
            max_players=2,
            fog_of_war=False,
            max_turns=100,
        )

    def random_action_fn(state, valid_mask, worker_id, rng):
        valid_actions = np.where(valid_mask)[0]
        if len(valid_actions) > 0:
            return int(rng.choice(list(valid_actions)))
        return 0

    pool = ParallelEnvPool(
        num_envs=num_envs,
        env_factory=env_factory,
        action_fn=random_action_fn,
        replay_buffer=buffer,
        max_steps_per_episode=50,
        seed=42,
    )

    start = time.time()
    pool.start()

    try:
        while pool.total_episodes < target_episodes:
            if time.time() - start > timeout_s:
                print(f"  ✗ Timeout: only {pool.total_episodes} episodes in {timeout_s}s")
                return False
            if pool.alive_workers == 0:
                print("  ✗ All workers died")
                return False
            time.sleep(0.5)
    finally:
        pool.stop(join_timeout=10.0)

    elapsed = time.time() - start
    steps = pool.total_env_steps
    print(f"  ✓ {pool.total_episodes} episodes, {steps} steps in {elapsed:.1f}s "
          f"({steps / elapsed:.1f} steps/s)")

    # Workers stopped
    assert pool.alive_workers == 0, f"Expected 0 alive workers after stop, got {pool.alive_workers}"
    print("  ✓ All workers stopped cleanly")

    # Buffer contents
    assert len(buffer) > 0, "Buffer is empty"
    batch = buffer.sample(min(32, len(buffer)))
    for state, action, reward, next_state, done in batch:
        assert state.shape == (9, board_size, board_size), f"Bad state shape: {state.shape}"
        assert next_state.shape == (9, board_size, board_size), f"Bad next_state shape: {next_state.shape}"
        assert isinstance(action, int), f"Bad action type: {type(action)}"
        assert isinstance(done, (bool, np.bool_)), f"Bad done type: {type(done)}"
    print(f"  ✓ Sampled {len(batch)} transitions with valid shapes/types")

    # Episode results drained correctly
    results = pool.pop_episode_results()
    assert len(results) >= target_episodes, f"Expected >= {target_episodes} results, got {len(results)}"
    worker_ids = {wid for _, _, wid, _ in results}
    outcomes = {outcome for _, _, _, outcome in results}
    assert outcomes <= {"win", "loss", "draw"}, f"Unexpected outcomes: {outcomes}"
    print(f"  ✓ {len(results)} episode results from workers {sorted(worker_ids)}, outcomes {sorted(outcomes)}")
    assert pool.pop_episode_results() == [], "Results not drained"

    return True


if __name__ == "__main__":
    print("Parallel Environment Pool Test Suite")
    print("=" * 50)

    # Buffer test needs no server
    if not test_replay_buffer_thread_safety():
        sys.exit(1)

    # Check if server is running
    import grpc
    from generals_pb.game.v1 import game_pb2, game_pb2_grpc

    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(width=5, height=5, max_players=2)
        ))
        channel.close()

        print("✓ Game server is running")

        success = test_parallel_env_pool()
        print("\n" + ("✓ All tests passed" if success else "✗ Tests failed"))
        sys.exit(0 if success else 1)

    except grpc.RpcError:
        print("\n✗ Error: Game server is not running!")
        print("  Please start the server with: go run cmd/game_server/main.go")
        sys.exit(1)
