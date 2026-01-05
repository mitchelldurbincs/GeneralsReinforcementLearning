# Services

This document reflects the current protobuf definitions under `proto/`.

## GameService (`proto/game/v1/game.proto`)

RPCs:
- `CreateGame`: optional `GameConfig` (width, height, max_players, fog_of_war, turn_time_ms, collect_experiences)
- `JoinGame`: `game_id`, `player_name`, optional `player_token`
- `SubmitAction`: `game_id`, `player_id`, `player_token`, `action`, optional `idempotency_key`
- `GetGameState`: `game_id`, `player_id`, `player_token` -> `GameState` with fog of war applied
- `StreamGame`: `game_id`, `player_id`, `player_token` -> stream of `GameUpdate`

Key messages:
- `GameState`: `turn`, `board`, `players`, `winner_id`, `action_mask`, `current_phase`
- `GameStateDelta`: incremental tile/player updates
- `GameUpdate`: one of `full_state`, `delta`, or `event`
- `GameEvent`: player join/elimination, game start/end, phase changes, disconnect/reconnect

## ExperienceService (`proto/experience/v1/experience.proto`)

RPCs:
- `StreamExperiences`: streams `Experience` entries; supports `game_ids`, `player_ids`, `batch_size`, `follow`
- `StreamExperienceBatches`: streams `ExperienceBatch` (defined in proto; not implemented in the Go server yet)
- `SubmitExperiences`: submit a batch of experiences
- `GetExperienceStats`: aggregate stats across buffers

Implementation notes:
- The Go server currently honors `game_ids`, `player_ids`, `batch_size`, and `follow` in `StreamExperiences`.
- `min_turn`, `enable_compression`, and `max_batch_wait_ms` are currently ignored by the server.
- `GetExperienceStats` currently populates `total_experiences` and `total_games` only.
