## Generals.io RL Edition
### 1. Solidify Core Game & Initial Tooling 🎮🧪🐳

This phase focuses on finalizing the game engine, ensuring its robustness through comprehensive testing, and getting basic containerization in place.

* **Continue Fleshing Out Game Logic & Features:**
    * Ensure all core mechanics (movement, combat, capture, production) are behaving as expected.
    * Finalize map generation parameters and ensure deterministic outputs with seeds.
    * Refine board display and visualization (color-coding, formatting) for debugging and potential UI use.
* **Comprehensive Unit Testing (Critical Priority):**
    * **Core Engine (`internal/game`):** Write extensive unit tests for actions, movement validation, combat resolution, production, state transitions, and error handling (custom error types).
        * Use "golden replays" (fixed seeds and action sequences) to assert final states.
    * **Map Generation (`mapgen`):** Test with deterministic seeds for reproducible maps.
    * **Performance Benchmarks:** Basic benchmarks for action processing.
* **Initial Docker Integration:**
    * Create `Dockerfile.go_server` for the Go game server.
    * Ensure the server can be built and run within a Docker container locally.
    * Future: Store images in AWS ECR.

---

### 2. Implement gRPC API & Game Server 📡⚙️

Develop the primary communication interface and the server that orchestrates game instances.

* **Define gRPC Protocols (`.proto` files in `api/`):**
    * **`GameService`:**
    * *Future Services (to be detailed later): `ReplayService`, `MatchMakerService`, `ModelService`.*
* **Develop Game Server (`internal/server`):**
    * Implement gRPC handlers for `GameService` (`grpc_handler.go`).
    * Manage game lifecycles (create, join, start, step, end).
    * Consider one goroutine per active game instance.
    * Use channel patterns for internal event handling (e.g., `actionQueue`, `stateUpdates`) if beneficial.
    * Translate between gRPC types and internal game engine types.
* **Basic Local Testing:**
    * Test gRPC endpoints with simple CLI clients (e.g., `grpcurl`) or a basic Go test client.
    * Ensure the server can run locally, handling game creation and action submission.

### 3. Establish Robust Logging & Replay System 📝💾

Implement comprehensive logging across the system and the ability to record and retrieve game replays, crucial for debugging and RL.

* **Structured Logging Framework (High Priority):**
    * **Go Server (`log/slog`):** Output structured JSON logs.
        * Log key events: `game_id_created`, `game_id_started`, `player_id_joined`, `action_submitted` (with player & game ID), `game_id_ended` (with winner/reason), critical errors.
        * Log basic performance metrics: `cpu_utilization` (if easily accessible).
    * **Python (RL side - later):** Consistent structured logging.
    * *Philosophy: Fail Fast, Log Everything.*
* **Game Replay System:**
    * **Storage:** Plan to store complete game histories (initial state, sequence of actions, potentially periodic full states) in S3.
    * **Functionality:**
        * Server-side logic to capture all actions and state changes for a game.
        * `GameService.GetReplay` RPC to retrieve replay data.
    * **`ReplayService` (gRPC - define `.proto`):**
        * `RecordExperience(RecordExperienceRequest) returns (RecordExperienceResponse)`
            * `Experience` entity: `bytes state` (observation), `bytes action_taken`, `float reward_received`, `bytes next_state` (next observation), `bool done_flag`, `string agent_version_id`, `float initial_td_error_priority` (optional).
        * `GetExperienceBatch(GetExperienceBatchRequest) returns (GetExperienceBatchResponse)`
    * *This service will be central for RL training data.*

### 4. Develop Python RL Integration & Local Training Loop 🤖🐍

Create the Python environment for the RL agent and enable a basic training loop running locally.

* **Python gRPC Client:**
    * Generate Python gRPC stubs from `.proto` files.
    * Create a Python client (`python/rl_agent/grpc_client.py`) to interact with the Go game server.
* **Gymnasium Environment Wrapper (`python/rl_agent/envs/`):**
    * Create `GeneralsEnv(gymnasium.Env)`.
    * `reset()`:
        * Calls `GameService.CreateGame` (or a dedicated "start new game for RL" RPC).
        * Receives initial `gamepb.GameState`.
        * Converts `gamepb.GameState` (especially `map_data` and `player_states`) into a NumPy array observation.
    * `step(action)`:
        * Converts Python RL action into `gamepb.Action`.
        * Calls `GameService.SubmitAction`. This gRPC call should block until the server processes the action and the next game tick, returning:
            * Next `gamepb.GameState` (for the next observation).
            * `float reward`.
            * `bool done`.
        * Calculates reward based on the response (see Reward Function).
        * Returns `observation`, `reward`, `done`, `truncated`, `info`.
    * **Observation Space Design:**
        * Board state as multi-channel tensor (e.g., owner, army count, tile type, visibility).
        * Player statistics (own army, land; potentially enemy stats if fully observable).
        * Turn number.
    * **Action Space Design:** Map directly to `MoveAction` parameters.
* **Define Reward Function (WIP, iterate on this):**
    * Win Game: `+100`
    * Lose Game: `-100`
    * Capture Enemy General: `+20`
    * Net Army Change (Self, Per Step): `+0.01 * (current_agent_army - previous_agent_army)`
    * Capture Enemy Standard Tile (Per Tile, Per Step): `+0.5`
    * Capture Enemy City (Per City, Per Step): `+2.0`
    * Per-Turn Time Penalty (Per Step): `-0.1`
    * *Consider sparse rewards initially, then add shaping.*
* **Basic RL Agent & Local Training:**
    * Choose an initial RL library (e.g., Stable Baselines3).
    * Implement a basic agent (e.g., DQN with a CNN if using visual input).
    * **Neural Network Design:** Start with a CNN for board state. Explore Transformers later, especially with Fog of War.
    * Set up a local training loop: Python agent interacts with the Go server running locally.
    * **Proof of Concept:** Get the agent to learn a very simple task (e.g., expand, survive for X turns).
    * **Metrics:** Log `average_episode_reward`, `average_episode_length`, `training_steps_per_second`.
    * *Python Coding Philosophy: Vectorize/batch operations where possible.*

---

### 5. Implement Fog of War & Ebiten UI 🌫️🎨

Introduce partial observability and create a UI for visualization and human interaction.

* **Fog of War (FoW) System (`internal/visibility`):**
    * Develop the logic to determine visibility for each player.
    * Integrate FoW into the `GameState` sent to players/agents (e.g., update `visible_map_data` in `PlayerState`).
    * Ensure the core engine can still manage the true full state while providing partial views.
    * Make FoW configurable.
* **Ebitengine UI Client (`cmd/ui_client` - Optional Go client, or other UI tech):**
    * Develop a client that can connect to the `GameService` (likely via `StreamGameState`).
    * Render the game board, units, and player stats.
    * Allow a human player to submit moves.
* **Enable Human vs. Bot Gameplay:**
    * Modify the system so the UI client can act as one player and an RL agent (running locally or via server) can be the opponent. This is key for evaluation and debugging.

---

### 6. Advanced RL: Self-Play, Scaling Prep & Infrastructure 🚀☁️

Focus on improving the agent through more advanced techniques and prepare the groundwork for larger-scale training and deployment.

* **Self-Play Strategies:**
    * **Agent Snapshots:** Save agent models at different training stages to use as opponents.
    * **Tournament System:** Periodically evaluate agents against each other. This might require a simple `MatchMakerService`.
        * **`MatchMakerService` (gRPC - define `.proto`):**
            * `FindMatch(FindMatchRequest) returns (FindMatchResponse)` (for general matchmaking if needed later)
            * `FindTournamentMatch(FindTournamentMatchRequest) returns (FindTournamentMatchResponse)` (to pair agents for evaluation)
    * **Exploiter Agents (Advanced):** Train agents specifically to beat current best agents.
    * **Prioritized Experience Replay (PER):** Use the `initial_td_error_priority` in the `Experience` entity with the `ReplayService`. Remove invalid/corrupt games from replay buffer.
    * **Map Variety:** Utilize `mapgen` to train on a diverse set of maps.
* **Complexity Progression for Agent Training:**
    * Start with smaller maps, full visibility.
    * Gradually increase map size, introduce Fog of War, more complex scenarios.
* **Containerization for RL:**
    * Create `Dockerfile.python_rl` for the Python training environment.
* **CI/CD Pipeline (GitHub Actions):**
    * Automate linting (e.g., `golangci-lint`), testing (Go & Python), coverage reports.
    * Automate Docker image builds (Go server, Python RL) and pushes to AWS ECR.
    * Run integration tests between Go server and Python client.
* **Cloud Infrastructure Planning (AWS):**
    * **Compute:**
        * Game Server (Go): CPU-optimized (e.g., EC2 M-series, C-series on ECS/EKS).
        * RL Learner (Python): GPU-optimized (e.g., EC2 P-series, G-series).
        * RL Actors (Python, for distributed training): CPU-optimized (e.g., EC2 C-series).
        * **Utilize Spot Instances** to reduce costs.
    * **Storage (S3):** Game replays, trained models, datasets, logs.
    * **Networking:** VPCs, security groups.
    * **Orchestration:** AWS ECS or EKS for game servers and potentially RL actors.
    * **Monitoring:** CloudWatch for basic infrastructure metrics, logs.
* **Model Management:**
    * **`ModelService` (gRPC - define `.proto`):**
        * `GetPolicy(GetPolicyRequest) returns (GetPolicyResponse)` (for actors to fetch latest model weights)
        * `PublishPolicy(PublishPolicyRequest) returns (PublishPolicyResponse)` (for learner to publish new models)

---

### 7. Distributed Training & Production Operations 🌍📈

Scale up RL training using distributed architectures and refine operational aspects.

* **Distributed RL Training Architecture:**
    * **Parallel Actors:** Multiple Python processes/containers running `GeneralsEnv`, generating experiences by interacting with game server instances.
        * Actors query the `ModelService` for the latest policy weights.
        * Actors send experiences to the central `ReplayService`.
    * **Centralized Learners:** One or more learner processes fetching mini-batches from the `ReplayService` and updating the model on GPUs.
        * Learners publish updated models via the `ModelService`.
* **Single/Multi-GPU Cloud Training:**
    * Deploy the learner component to EC2 GPU instances.
    * Scale out actors across CPU instances.
* **Enhanced Observability & Monitoring (Production Grade):**
    * **Metrics:**
        * Integrate Prometheus for metrics collection from Go server and Python components.
        * Key Go server metrics: Games per second, active games, action processing latency, error rates.
        * Key RL metrics: `agent_elo_ratings`, `replay_buffer_size`, `training_steps_per_second`, `average_episode_reward/length` over time.
        * Infrastructure metrics: `container_resource_usage`, `network_throughput`, `storage_iops`.
    * **Tracing (OpenTelemetry):** Implement for tracing requests across gRPC services (e.g., client -> game server -> replay service).
    * **Health Probes:** Implement `/health/live` and `/health/ready` HTTP endpoints in the Go server for orchestrators.
* **Performance Profiling & Optimization:**
    * Continuously profile Go server and Python RL pipeline.
    * Optimize bottlenecks in game logic, data processing, or network communication.
* **Success Metrics Re-evaluation:**
    * Correctness: 100% deterministic game outcomes with identical seeds.
    * Performance: Aim for target simultaneous games on defined hardware.
    * Reliability: Uptime targets for training workloads.
    * RL Training: Convergence speed, final agent skill level.