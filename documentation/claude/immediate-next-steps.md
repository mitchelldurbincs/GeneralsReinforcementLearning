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

- [ ] Run `python/train_dqn_simple.py` for a handful of episodes on a small
      board. Success = no crashes, loss values printed, episodes complete.
- [ ] Verify checkpoint save/load round-trips (`train_dqn_agent.py` has
      `save_model()`/`load_model()`).
- [ ] Triage the ~15 stale `python/test_*.py` debug scripts from past
      iterations: delete or move the dead ones so the directory stays
      navigable. Keep `test_gym_env.py`, `test_gym_minimal.py`,
      `test_grpc_client.py`.
- [ ] Record a baseline here: average episode reward of the untrained/early
      agent vs the random opponent, episode length, episodes/minute.

## Day 3 — Stabilize and pick the next milestone

- [ ] Longer run with `python/train_dqn_robust.py` (a few hundred episodes).
      Note any failure modes — stream reconnection and buffer cleanup on game
      end are known weak spots (see CLAUDE.md TODOs).
- [ ] Update CLAUDE.md "Development Status" to match reality: Gym env is done,
      proto scripts work (if Day 1 confirms), DQN loop status, and remove
      fixed items from "TODO/Known Issues".
- [ ] Decide the next milestone and start a plan doc for it:
      - **Option A:** Multi-game parallel experience collection (Phase 1 of
        `training-next-steps.md`) — faster training.
      - **Option B:** Wire `cmd/ui_client` to the gRPC server so a human can
        play against Python agents / trained models — currently the UI only
        supports a local random AI and has zero gRPC integration (~500+ LOC).

## Baseline Metrics

(Fill in on Day 2/3.)

| Metric | Value | Notes |
|--------|-------|-------|
| Avg episode reward (early agent vs random) | | |
| Avg episode length (turns) | | |
| Training throughput (episodes/min) | | |
| Crash-free episodes in long run | | |
