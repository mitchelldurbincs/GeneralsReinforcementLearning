# Immediate Next Steps: Getting DQN Training Running End-to-End

**Created:** 2026-06-09
**Scope:** The next few days of work. Goal: prove the training pipeline runs
end-to-end (server → Gym env → DQN training loop → checkpoints), even if the
agent learns nothing impressive yet. Distributed training, self-play, and
human-vs-trained-agent play are explicitly out of scope here (see
`training-next-steps.md` for the longer roadmap).

## Current State (verified 2026-06-09)

**What works:**
- Core Go engine, events, state machine, mapgen: all tests pass.
- Experience streaming backend (`stream_aggregator.go`, `batch_processor.go`,
  `experience_service.go`) implemented with integration tests.
- `python/generals_gym/generals_env.py` is **complete** — `reset()`, `step()`,
  reward calculation, random opponent handling, action masking, `close()`, and
  Gym registration (`Generals-v0`) are all implemented. It connects to the gRPC
  game server. (Older docs claiming it's unfinished are stale.)
- UI client (`cmd/ui_client`) is current: human vs local random AI works.

**What's broken / unverified:**
- Hardcoded dev-machine paths (`/home/aspect/source/...`) in 5 Python scripts:
  `train_dqn_simple.py:16`, `train_dqn_agent.py:23`, `train_dqn_robust.py:20`,
  `test_gym_env.py:13`, `test_gym_minimal.py:5`.
- `python/requirements.txt` is missing `torch` and `gymnasium`.
- Generated Go protos are gitignored and must be regenerated on a fresh clone
  (`make generate-protos` → `pkg/api/`, currently blocks
  `go test ./internal/experience/...` and `./internal/grpc/...`). Python protos
  (`python/generals_pb/`) are actually committed to git and verified fresh on
  Day 1 — regenerating only churns the generator version comment.
- None of the three DQN training scripts have been verified to run end-to-end.

## Day 1 — Make the pipeline runnable

- [x] Fix hardcoded `sys.path.insert(0, '/home/aspect/...')` in the 5 scripts
      listed above; use a path relative to the script, e.g.
      `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`.
- [x] Add `torch>=2.0.0` and `gymnasium>=0.29.0` to `python/requirements.txt`
      (optionally `tensorboard` for training curves). Also fixed
      `scripts/generate-python-protos.sh`, which unconditionally overwrote
      `requirements.txt` with a stale dependency list — it now only creates
      the file if missing.
- [x] Regenerate protos and verify:
      - `make generate-protos`, then `go build ./...` and `go test ./...`
        (UI packages may still fail without X11 headers — that's fine headless).
        All non-UI tests pass. Two fixes were needed in
        `internal/experience`: `TestEnhancedCollector_OverflowPersist` used a
        fixed 200ms sleep (flaky under load; now polls with
        `require.Eventually`), and `EnhancedCollector`'s batch processor could
        drop in-flight experiences on `Close()` because `ctx.Done()` raced the
        stream channel in its `select` — it now drains the stream channel
        before the final flush (this was the
        `TestEnhancedCollector_LoadFromPersistence` flake).
      - `./scripts/generate-python-protos.sh`, then verify
        `from generals_pb.game.v1 import game_pb2` imports cleanly from
        `python/` with the venv active. Committed protos were already fresh.
- [x] Smoke test the Gym env: start the server
      (`go run cmd/game_server/main.go`), then from `python/`:
      `python -c "from generals_gym import GeneralsEnv; env = GeneralsEnv(); obs, info = env.reset(); print(obs.shape)"`
      followed by a few random `env.step(...)` calls.
      Works: observation shape `(9, 15, 15)`, 5 random steps OK.
- [x] Run `python/test_gym_minimal.py` and `python/test_gym_env.py`; fix
      whatever surfaces. Both pass against a live server with no further
      fixes needed.

## Day 2 — Verify training end-to-end

- [x] Run `python/train_dqn_simple.py` for a handful of episodes on a small
      board. Success = no crashes, loss values printed, episodes complete.
      Done: 20/20 episodes complete on a 5x5 board, loss decreasing
      (0.015 → 0.003). Two issues fixed along the way:
      - **Map generation failed randomly on small boards**: with the default
        `MinGeneralSpacing` of 5, a first general placed near the center of a
        5x5 board leaves no tile ≥5 Manhattan away, so `CreateGame`
        intermittently returned Internal errors (hit on episode 14 of the
        first run). `DefaultMapConfig` now clamps spacing to `w/2 + h/2`
        (test added in `mapgen/generator_test.go`).
      - `train_dqn_simple.py` never printed loss values despite that being
        the success criterion — it now prints per-episode average loss.
- [x] Verify checkpoint save/load round-trips (`train_dqn_agent.py` has
      `save_model()`/`load_model()`). Verified with a one-off script: trained
      a `DQNAgent` for 2 episodes, saved, loaded into a freshly-initialized
      agent, and confirmed q/target network weights, Adam optimizer state,
      epsilon, and step/episode counters are all bit-identical, plus an
      identical forward pass on a fixed input.
- [x] Triage the ~15 stale `python/test_*.py` debug scripts from past
      iterations: delete or move the dead ones so the directory stays
      navigable. Keep `test_gym_env.py`, `test_gym_minimal.py`,
      `test_grpc_client.py`. Deleted 17 one-off debug scripts (streaming
      experiments, board-state dumps, import checks); none were referenced
      by docs, Makefile, CI, or other code.
- [x] Record a baseline here: average episode reward of the untrained/early
      agent vs the random opponent, episode length, episodes/minute.
      See Baseline Metrics table below.

## Day 3 — Stabilize and pick the next milestone

- [x] Longer run with `python/train_dqn_robust.py` (a few hundred episodes).
      Done: 300 episodes on a 10x10 board, 200-turn cap. Findings:
      - **Failure mode confirmed — server game-lifecycle gap.** The first
        300-episode attempt died at episode 94 with `CreateGame` failing
        ("server at capacity: 100/100"). Cause: the Gym env truncates
        episodes client-side, but there is no `DeleteGame`/`LeaveGame` RPC
        and no `max_turns` in `GameConfig`, so every truncated game sits in
        `PhaseRunning` until the server's 30-minute `abandonedGameTimeout` —
        at ~4 games/min that crosses `max_games=100` mid-run. Client-side
        fixes applied: `GeneralsEnv._connect_to_server` no longer creates a
        leaked probe game (uses `channel_ready_future`), and the trainer's
        `reset_environment` now backs off exponentially (~10 min total)
        instead of dying after 3×2 s. Server-side fix (the real one) is the
        first task of the next milestone. Rerun with the server started as
        `--max-games 0`: **300/300 episodes, 60k steps, 74.4 min, zero
        errors/warnings, zero capacity rejections.**
      - **Buffer cleanup (CLAUDE.md weak spot):** no memory problem observed
        — server RSS stayed ~23-25 MB across 300 created games while the
        cleanup ticker reaped 180 of them mid-run. The issue is reaping
        *latency* (30 min for abandoned games) against the game-count cap,
        not a leak. Note the Gym path never sets `collect_experiences`, so
        the experience-buffer side of this TODO is still unexercised.
      - **Stream reconnection (CLAUDE.md weak spot):** not exercised — the
        Gym training path uses CreateGame/JoinGame/SubmitAction/GetGameState
        only, no experience streaming. Still an open TODO.
      - **DQN loss divergence:** with MSE loss, average loss exploded
        monotonically (0.0125 at ep 10 → 5.8e3 at ep 50 → 8.4e14 at ep 300)
        while reward stayed flat (~1.2/episode) — classic Q-overestimation
        with max-bootstrapping over 500 actions and unscaled shaping
        rewards. Switched to Huber (`smooth_l1`) loss; a 30-episode
        validation run keeps loss in single digits (1.86 at ep 30 vs ~1e3+
        for MSE at that point), though it still trends up — reward
        scaling/clipping and Double DQN are TODO for the next milestone.
      - **Play strength:** every one of the 300 episodes hit the 200-step
        cap; no wins/losses ever occurred, so reward is pure shaping noise.
        Throughput ~4 episodes/min (10x10, 200 steps; gRPC + the env's
        50 ms/step sleep dominate). Real learning signal needs decisive
        games (server-side `max_turns` with a tile-count winner) and more
        throughput (parallel envs) — both in the next milestone.
- [x] Update CLAUDE.md "Development Status" to match reality: Gym env is done,
      proto scripts work (if Day 1 confirms), DQN loop status, and remove
      fixed items from "TODO/Known Issues". Done: Gym env, proto generation,
      and single-env DQN loop moved to Completed; stale proto-import TODO
      removed; game-lifecycle gap added as a Known Issue.
- [x] Decide the next milestone and start a plan doc for it:
      - **Option A:** Multi-game parallel experience collection (Phase 1 of
        `training-next-steps.md`) — faster training.
      - **Option B:** Wire `cmd/ui_client` to the gRPC server so a human can
        play against Python agents / trained models — currently the UI only
        supports a local random AI and has zero gRPC integration (~500+ LOC).
      **Chose Option A** — see `documentation/claude/next-milestone.md`.
      Training works end-to-end but is throughput-bound (~4 eps/min) and the
      Day 3 run showed the server-side lifecycle work A requires is needed
      for *any* long run. B is the natural follow-up once there's a trained
      agent worth playing against.

## Baseline Metrics

Measured 2026-06-09 with `train_dqn_simple.py` (20 episodes, 5x5 board, no
fog, vs the env's built-in random opponent, CPU-only torch 2.12, epsilon
0.82–1.0 so the agent is still mostly random).

| Metric | Value | Notes |
|--------|-------|-------|
| Avg episode reward (early agent vs random) | 1.11 | Nearly constant across episodes — dominated by per-step shaping; no wins/losses observed |
| Avg episode length (turns) | 100.0 | Every episode hit the script's 100-step cap; no game reached a decisive end on 5x5 in 100 steps |
| Training throughput (episodes/min) | ~9.9 | 20 episodes in 2m02s, single env, CPU; step latency is gRPC round-trip bound, not compute bound |
| Crash-free episodes in long run | 42/42 | 2×20-episode runs + 2 episodes for the checkpoint test, zero crashes (after the mapgen spacing fix) |

Caveat: with every episode truncated at the step cap, reward/length say
little about play strength yet — the Day 3 longer run on a bigger board
with a higher cap will give a more meaningful baseline.
