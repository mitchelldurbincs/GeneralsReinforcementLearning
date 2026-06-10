#!/usr/bin/env python3
"""
Verify that games end decisively (general capture) and that the win/loss
terminal transition lands in the replay buffer with reward +/-100 and
done=True. Requires a running game server built with GameConfig.max_turns
support (server-authoritative truncation).

Phase 2 of documentation/claude/parallel-experience-collection-plan.md.
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generals_gym import GeneralsEnv, ParallelEnvPool, ReplayBuffer

BOARD_SIZE = 6
MAX_TURNS = 800
NUM_ENVS = 4
TARGET_EPISODES = 8
MIN_DECISIVE = 3
TIMEOUT_S = 420
WIN_REWARD = 100.0


def test_decisive_games():
    """Random-vs-random pool on a small board: most games should end by
    general capture, and those terminal transitions must carry the win/loss
    reward with done=True."""
    print(f"\nCollecting {TARGET_EPISODES} random-vs-random episodes "
          f"on {BOARD_SIZE}x{BOARD_SIZE} (max_turns={MAX_TURNS})...")

    buffer = ReplayBuffer(capacity=50000)

    def env_factory(worker_id):
        return GeneralsEnv(
            server_address="localhost:50051",
            board_width=BOARD_SIZE,
            board_height=BOARD_SIZE,
            max_players=2,
            fog_of_war=True,
            max_turns=MAX_TURNS,
        )

    def random_action_fn(state, valid_mask, worker_id, rng):
        valid_actions = np.where(valid_mask)[0]
        if len(valid_actions) > 0:
            return int(rng.choice(list(valid_actions)))
        return 0

    pool = ParallelEnvPool(
        num_envs=NUM_ENVS,
        env_factory=env_factory,
        action_fn=random_action_fn,
        replay_buffer=buffer,
        max_steps_per_episode=MAX_TURNS + 50,
        seed=123,
    )

    start = time.time()
    pool.start()
    try:
        while pool.total_episodes < TARGET_EPISODES:
            if time.time() - start > TIMEOUT_S:
                print(f"  ✗ Timeout: only {pool.total_episodes} episodes in {TIMEOUT_S}s")
                return False
            if pool.alive_workers == 0:
                print("  ✗ All workers died")
                return False
            time.sleep(0.5)
    finally:
        pool.stop(join_timeout=15.0)

    elapsed = time.time() - start
    print(f"  ✓ {pool.total_episodes} episodes, {pool.total_env_steps} steps "
          f"in {elapsed:.0f}s")

    # Snapshot the buffer (workers are stopped, no concurrent writes)
    transitions = list(buffer._buffer)

    terminals = [(i, r, d) for i, (s, a, r, ns, d) in enumerate(transitions)
                 if abs(r) >= WIN_REWARD]
    wins = sum(1 for _, r, _ in terminals if r >= WIN_REWARD)
    losses = sum(1 for _, r, _ in terminals if r <= -WIN_REWARD)
    print(f"  Terminal transitions in buffer: {len(terminals)} "
          f"({wins} wins, {losses} losses)")

    for i, r, d in terminals[:5]:
        print(f"    buffer[{i}]: reward={r:+.1f} done={d}")

    assert len(terminals) >= MIN_DECISIVE, (
        f"Expected >= {MIN_DECISIVE} decisive games, got {len(terminals)}. "
        f"Games are not ending by general capture.")

    bad = [(i, r, d) for i, r, d in terminals if not d]
    assert not bad, f"Terminal-reward transitions without done=True: {bad}"
    print(f"  ✓ All {len(terminals)} win/loss terminals have done=True")

    # done=True must also be rare overall (only episode ends), not every step
    done_count = sum(1 for s, a, r, ns, d in transitions if d)
    assert done_count <= pool.total_episodes + NUM_ENVS, (
        f"done=True on {done_count} transitions for {pool.total_episodes} episodes")
    print(f"  ✓ done=True on {done_count} transitions for "
          f"{pool.total_episodes} episodes")

    return True


if __name__ == "__main__":
    print("Decisive Games Test Suite")
    print("=" * 50)

    import grpc
    from generals_pb.game.v1 import game_pb2, game_pb2_grpc

    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        resp = stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(width=5, height=5, max_players=2,
                                       max_turns=100)
        ))
        channel.close()
        if resp.config.max_turns != 100:
            print("✗ Server does not support GameConfig.max_turns — rebuild it")
            sys.exit(1)
        print("✓ Game server is running (max_turns supported)")
    except grpc.RpcError:
        print("\n✗ Error: Game server is not running!")
        print("  Please start the server with: go run cmd/game_server/main.go")
        sys.exit(1)

    success = test_decisive_games()
    print("\n" + ("✓ All tests passed" if success else "✗ Tests failed"))
    sys.exit(0 if success else 1)
