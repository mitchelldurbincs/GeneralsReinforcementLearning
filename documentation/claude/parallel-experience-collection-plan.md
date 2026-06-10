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

- [ ] Add `python/generals_gym/vector_env.py`: a `ParallelEnvPool` that owns N
      `GeneralsEnv` instances, each stepped from its own worker thread running
      a full episode loop (reset → step* → done). Workers push
      `(state, action, reward, next_state, done)` into a shared queue/buffer.
      Handle per-env failures by recreating that env without killing the pool
      (reuse the retry pattern from `train_dqn_robust.py:reset_environment`).
- [ ] Add a thread-safe `ReplayBuffer` (promote the one in
      `train_dqn_agent.py`, add a lock; also closes the
      `create_replay_buffer()` NotImplementedError stub in
      `experience_consumer.py` or deletes it).
- [ ] Add `python/train_dqn_parallel.py` (or a `--num-envs N` flag on
      `train_dqn_robust.py`): main thread trains from the shared buffer at a
      fixed train-steps-per-env-step ratio; workers use the latest network for
      action selection with per-worker epsilon.
- [ ] Checkpointing parity: keep `train_dqn_robust.py`'s save/resume behavior
      (network, optimizer, epsilon, counters) in the parallel trainer.
- [ ] Benchmark: steps/s and episodes/min at N = 1, 4, 8, 16 on a 10x10 board.
      Record the scaling curve here. Investigate if scaling is sublinear
      before N=8 (likely suspects: server lock contention in game_manager,
      GIL pressure from tensor conversion).
- [ ] Stress the known weak spot: run N=8+ with
      `collect_experiences: true` for 100+ games and watch server RSS —
      the Day 3 run only verified flat memory with collection off.

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

(to be filled in as tasks complete)

| N envs | steps/s | eps/min | server RSS | notes |
|--------|---------|---------|------------|-------|
| 1 (baseline, Day 3) | ~12 | ~2.4 | ~22 MB | 10x10, 300-step cap |
