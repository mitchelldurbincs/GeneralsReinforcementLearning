# Parallel Experience Collection Plan

**Created:** 2026-06-10
**Status:** Not started
**Decision record:** This is the milestone chosen at the end of
`immediate-next-steps.md` Day 3 (Option A). Option B (wiring `cmd/ui_client`
to the gRPC server for human-vs-agent play) is deferred; nothing in this plan
blocks doing it later.

## Why this milestone

The Day 3 long run (300 episodes, 10x10, 2h06m) confirmed training is
**gRPC round-trip bound**: ~12 steps/s ≈ 2.4 episodes/min with a single env,
on CPU, with the network forward/backward pass nowhere near saturating the
machine. The server already supports 100 concurrent games and the
aggregation/batching backend exists. The missing piece is client-side
orchestration: N environments collecting in parallel, one learner consuming.

Target: **8–16x throughput** (roughly linear scaling to 8–16 parallel envs),
i.e. hundreds of episodes per hour instead of ~140.

## Design sketch

Phase 1 stays entirely in Python and uses the existing per-env gRPC path
(each `GeneralsEnv` already creates its own game on the server). Server-side
experience streaming (`StreamExperienceBatches`) is deliberately *not* used
yet — it decouples acting from learning and is better tackled once the
simple version works.

```
N x GeneralsEnv (threads)  ──>  shared ReplayBuffer  ──>  single DQN learner
        ^                                                      |
        └────────────── updated policy (epsilon-greedy) ───────┘
```

Threads are sufficient: gRPC calls release the GIL, and the env step is
I/O-bound. Each worker thread runs its own episode loop with the current
policy network (shared, read-only forward passes under `torch.no_grad()`),
pushes transitions into a thread-safe replay buffer, and the main thread
runs the training loop.

## Tasks

### Phase 1: parallel collection, single learner

- [x] Add `python/generals_gym/vector_env.py`: a `ParallelEnvPool` that owns N
      `GeneralsEnv` instances, each stepped from its own worker thread running
      a full episode loop (reset → step* → done). Workers push
      `(state, action, reward, next_state, done)` into a shared queue/buffer.
      Handle per-env failures by recreating that env without killing the pool
      (reuse the retry pattern from `train_dqn_robust.py:reset_environment`).
- [x] Add a thread-safe `ReplayBuffer` (promote the one in
      `train_dqn_agent.py`, add a lock; also closes the
      `create_replay_buffer()` NotImplementedError stub in
      `experience_consumer.py` or deletes it).
      → `python/generals_gym/replay_buffer.py` (list-based ring buffer +
      lock; `total_pushed` doubles as the global env-step counter); the
      stub was deleted. Smoke-tested via `python/test_parallel_env.py`.
- [x] Add `python/train_dqn_parallel.py` (or a `--num-envs N` flag on
      `train_dqn_robust.py`): main thread trains from the shared buffer at a
      fixed train-steps-per-env-step ratio; workers use the latest network for
      action selection with per-worker epsilon.
      → New script (robust stays as the N=1 baseline). Workers run their own
      `torch.no_grad()` forward passes on the shared network; fixed Ape-X
      style per-worker epsilons (0.4^(1+7·i/(N−1))); `--train-ratio` paces
      the learner on run-local counters (0 = pure-collection benchmark).
- [x] Checkpointing parity: keep `train_dqn_robust.py`'s save/resume behavior
      (network, optimizer, epsilon, counters) in the parallel trainer.
      → Same checkpoint keys plus `learner_steps`; resume round-trip
      verified (episodes/env-steps continue, pacing doesn't inherit the
      previous run's deficit).
- [x] Benchmark: steps/s and episodes/min at N = 1, 4, 8, 16 on a 10x10 board.
      Record the scaling curve here. Investigate if scaling is sublinear
      before N=8 (likely suspects: server lock contention in game_manager,
      GIL pressure from tensor conversion).
      → Collection scales near-linearly to N=16 (see table). The component
      that degrades is the **learner**: gradient steps/s fall from 22 (N=4)
      to 13 (N=16) as worker threads steal the GIL during observation
      building, so the achieved train ratio collapses (0.32 → 0.05). On CPU,
      N=4–8 is the sweet spot for actual training; N=16 is only worth it for
      pure collection or with a GPU learner.
- [x] Stress the known weak spot: run N=8+ with
      `collect_experiences: true` for 100+ games and watch server RSS —
      the Day 3 run only verified flat memory with collection off.
      → N=8, 110 collected games, full throughput maintained (135 steps/s,
      no penalty vs 138 without collection). RSS grew 51 → 86 MB during the
      run (~320 KB/game) and kept growing (~100 MB) after the client
      disconnected. Root cause characterized, **no unbounded leak**:
      - Server games **never reach ENDED**: the proto's `GameConfig` has no
        `max_turns`, so the env truncates client-side and abandons the game
        mid-Running. The `finishedGameTTL` cleanup path is dead code for RL
        traffic.
      - Abandoned games keep self-advancing turns (`turn_time_ms=500` arms
        server turn timers), burning ~12% CPU for ~700 zombie games and —
        with collection on — filling their per-game experience buffers
        (10k cap) long after the client left.
      - The abandoned-game path does reap them: `lastActivity` only updates
        on JoinGame/SubmitAction, so 30 min after the last client action the
        5-min cleanup tick removes them (observed: `cleaned=117
        remaining=649` at the first eligible tick).
      - Steady-state cost of continuous N=8 training is therefore ~30 min ×
        ~40 games/min ≈ 1,200 zombie games. Workable for now; the real fix
        is server-side and shared with the max_games issue below: add
        `max_turns` to GameConfig (games actually finish) and/or a
        DeleteGame RPC called from `GeneralsEnv.reset()`.

### Phase 2: make the runs meaningful

- [ ] Decisive games: with throughput fixed, raise caps (e.g. 1000 turns) so
      win/loss rewards actually appear; verify the terminal reward shows up
      in collected transitions. If random-vs-random still never terminates,
      consider shrinking to 8x8 or seeding asymmetric starts for training.
- [ ] Re-baseline: win rate vs random opponent after a multi-thousand-episode
      run; record metrics table here (reward, episode length, win rate,
      eps/min, server RSS).

### Phase 3 (stretch, separate plan if it grows)

- [ ] Evaluate switching collection to server-side
      `StreamExperienceBatches` so acting and learning fully decouple
      (prerequisite for distributed training).
- [ ] Prioritized replay.

## Benchmark results

2026-06-10, 10x10 board, 300-step cap (games end at the 200-turn server
cap), ~110s steady-state per run, CPU-only host. "ratio" is achieved
learner-gradient-steps per env step (`--train-ratio 1.0` requested).

| N envs | steps/s | eps/min | learner steps/s (ratio) | steps/s, train-ratio 0 | notes |
|--------|---------|---------|-------------------------|------------------------|-------|
| 1 (baseline, Day 3) | ~12 | ~2.4 | ~12 (1.00) | — | single-env `train_dqn_robust.py` |
| 1  | 18.4  | 6.0  | 18.4 (1.00) | 18.3  | parallel harness, learner keeps up |
| 4  | 69.8  | 17.9 | 22.1 (0.32) | 72.1  | 5.8x baseline |
| 8  | 138.3 | 47.9 | 19.3 (0.14) | 140.5 | 11.5x baseline |
| 16 | 250.8 | 83.7 | 13.2 (0.05) | 273.2 | 20.9x baseline; 96 eps/min at ratio 0 |

Server RSS grew from ~21 MB to ~47 MB across the whole sweep (~470 games
created, collection off) — that growth is abandoned games awaiting the
30-minute reaper (see the stress-test findings above: server games never
reach ENDED, so the abandoned path is the one that fires), not a leak.

Takeaways:

- **Target met**: 11.5–21x throughput vs the Day 3 baseline; collection
  scaling is near-linear to N=16, so the server is not the bottleneck.
- **The learner is the new bottleneck on CPU**: worker GIL pressure cuts
  gradient throughput as N rises. For training (not just collection) use
  N=4–8, or lower `--train-ratio`, or move the learner to GPU.
- **Server-side blocker found and worked around**: there is no DeleteGame
  RPC and RL games are only reaped as *abandoned* 30 min after the last
  client action (`internal/grpc/gameserver/server.go:42-44`; games never
  reach ENDED because GameConfig has no max_turns), so parallel rates
  exhaust the default `max_games=100` within minutes. Run the server with
  `--max-games 5000` for now; proper fix (max_turns in GameConfig and/or a
  DeleteGame RPC called from `GeneralsEnv.reset()`) is noted in CLAUDE.md
  known issues.
