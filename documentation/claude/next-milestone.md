# Next Milestone: Multi-Game Parallel Experience Collection (Option A)

**Created:** 2026-06-09 (Day 3)
**Decision:** Option A — multi-game parallel experience collection.
**Deferred:** Option B (wire `cmd/ui_client` to gRPC for human-vs-agent play).

## Why Option A

The Day 1–3 work proved the pipeline runs end-to-end: server → Gym env →
DQN training loop → checkpoints. The binding constraint is now **experience
generation throughput**, not correctness:

- Single-env training runs at ~7–10 episodes/min on a 10x10 board with a
  200-turn cap. Each `env.step()` is 3 sequential gRPC round-trips plus a
  hardcoded `time.sleep(0.05)` — a 200-step episode spends ≥10 s just
  sleeping. CPU is nearly idle; the GPU/learner side is starved.
- The Day 3 long run (300 episodes) exposed server-side game lifecycle gaps
  (see "Findings feeding into this milestone" below) that must be fixed for
  *any* serious training run — and they are exactly the prerequisites for
  running many games in parallel.
- Option B is ~500+ LOC of UI/gRPC integration whose payoff is being able to
  *play against* a trained agent. Until parallel collection produces an agent
  that beats a random opponent reliably, there is nothing interesting to play
  against. B becomes the natural milestone after A.

This is also Phase 1 ("Multi-Game Training, Weeks 1-2") of the long-term
roadmap in `training-next-steps.md`, so it keeps us on that path.

## Findings feeding into this milestone (Day 3 long run)

1. **Truncated games are never ended.** The Gym env truncates episodes
   client-side at `max_turns`, but the server has no `max_turns` in
   `GameConfig` and no `DeleteGame`/`LeaveGame` RPC, so every truncated game
   sits in `PhaseRunning` until the 30-minute `abandonedGameTimeout`. At
   ~4 games/min this crosses the default `max_games=100` cap mid-run and
   `CreateGame` starts failing with "server at capacity".
2. **The trainer's recovery is too shallow for capacity exhaustion.**
   `reset_environment()` retried 3× with 2 s sleeps; capacity exhaustion
   lasts minutes (until the cleanup ticker reaps old games), so the run
   died instead of stalling. Fixed on Day 3 with exponential backoff, but
   the real fix is (1).
3. **Connection probes leaked games.** `GeneralsEnv._connect_to_server()`
   verified connectivity by creating a real (never-joined, never-cleaned)
   game. Fixed on Day 3 (`channel_ready_future`), but worth remembering:
   with N parallel envs, small per-env leaks multiply by N.

## Scope

### 1. Server-side game lifecycle (prerequisite, Go)

- Add `max_turns` to `GameConfig` (proto + server): the engine ends the game
  (e.g. turn-limit win condition or a "truncated" end state) when the limit
  is reached, so games reach `PhaseEnded` and are reaped by the existing
  `finishedGameTTL` path instead of the 30-minute abandoned path.
- Add a `DeleteGame` (or `LeaveGame`) RPC so a client that abandons an
  episode early (env `close()`, trainer crash) can release the game
  immediately.
- Verify experience buffers and stream registrations are released on both
  paths (`CLAUDE.md` "buffer cleanup" TODO).

Key files:
- `proto/game/v1/game.proto`, regenerate via `make generate-protos` and
  `./scripts/generate-python-protos.sh`
- `internal/grpc/gameserver/server.go`, `game_manager.go`
- `internal/game/engine.go`, `internal/game/rules/win_conditions.go`
- `internal/game/states/` (turn-limit → Ending transition)

### 2. Vectorized environment (Python)

- Make `GeneralsEnv` safe for `gymnasium.vector.AsyncVectorEnv` (one env per
  worker process, no shared channel state) and pass `max_turns` through to
  `CreateGame` once (1) lands.
- Add `python/generals_gym/vector.py` with a small factory:
  `make_vec_env(n_envs, **env_kwargs)`.
- Remove/shrink the fixed `time.sleep(0.05)` in `step()` — poll
  `GetGameState` until the turn number advances, with a short timeout.
  (With many parallel envs the sleep matters less, but it still caps
  per-env throughput at ~20 steps/s.)

Key files:
- `python/generals_gym/generals_env.py`
- `python/generals_gym/vector.py` (new)

### 3. Batched DQN training loop (Python)

- New `python/train_dqn_parallel.py` (or extend `train_dqn_robust.py`):
  batched action selection (one forward pass for N observations), shared
  replay buffer fed by all envs, same checkpointing as the robust trainer.
- Keep per-env error recovery: one env crashing must not kill the other
  N-1 (AsyncVectorEnv restarts or per-env try/except in a custom loop).

Key files:
- `python/train_dqn_parallel.py` (new)
- `python/train_dqn_robust.py` (shared agent/network code could be factored
  into `python/generals_agent/dqn.py` if duplication gets noisy)

### 4. Measure

- Re-run the Day 2/3 baseline at N = 1, 4, 8, 16 envs; record
  episodes/min and steps/s in `immediate-next-steps.md`-style tables.
- Success criterion: ≥5x experience throughput at N=8 vs N=1, and a
  300-episode run with zero capacity errors.

## Estimated scope

- Server lifecycle work (proto + engine + gRPC + tests): ~2-3 days
- Vectorized env + parallel trainer: ~2 days
- Benchmarking + doc updates: ~half day

Roughly one week of focused work; each piece lands independently
(server `max_turns` is useful on its own even for single-env training).

## First concrete task

Add `max_turns` to `GameConfig` end-to-end: proto field → engine turn-limit
end condition (game transitions to `PhaseEnding`/`PhaseEnded`, winner =
player with most tiles, or draw) → pass-through in `GeneralsEnv.reset()` →
test that a truncated game is reaped via the `finishedGameTTL` path. This
single change removes the biggest operational hazard for long runs (capacity
exhaustion) and is a prerequisite for everything else in this milestone.
