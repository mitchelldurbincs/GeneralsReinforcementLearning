"""Parallel environment pool for multi-env experience collection.

Runs N GeneralsEnv instances, each confined to its own worker thread
(env instance state is not thread-safe, but each env owns its own gRPC
channel so threads never share connections). Workers run full episode
loops and push transitions into a shared thread-safe ReplayBuffer; gRPC
calls release the GIL, so threads are sufficient for the I/O-bound env
step (~80ms round trip).
"""

import logging
import random
import threading
import time
from typing import Any, Callable, List, Tuple

import numpy as np

from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

# (state, valid_mask, worker_id, rng) -> action index
ActionFn = Callable[[np.ndarray, np.ndarray, int, random.Random], int]


class ParallelEnvPool:
    """Owns N environments, each stepped from its own worker thread.

    Workers push (state, action, reward, next_state, done) into the shared
    replay buffer. A failed env is recreated in place (same retry pattern
    as train_dqn_robust.reset_environment) without killing the pool; a
    worker only dies after max_env_retries consecutive recreation failures.
    """

    def __init__(
        self,
        num_envs: int,
        env_factory: Callable[[int], Any],
        action_fn: ActionFn,
        replay_buffer: ReplayBuffer,
        max_steps_per_episode: int = 200,
        max_env_retries: int = 3,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.env_factory = env_factory
        self.action_fn = action_fn
        self.replay_buffer = replay_buffer
        self.max_steps_per_episode = max_steps_per_episode
        self.max_env_retries = max_env_retries
        self.seed = seed

        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._stats_lock = threading.Lock()
        self._total_episodes = 0
        self._alive_workers = 0
        # (episode_reward, episode_length, worker_id, outcome), drained by the
        # trainer; outcome is "win"/"loss" (decisive) or "draw" (truncated)
        self._episode_results: List[Tuple[float, int, int, str]] = []

    def start(self) -> None:
        """Spawn one daemon thread per environment."""
        if self._threads:
            raise RuntimeError("Pool already started")
        self._stop_event.clear()
        self._alive_workers = self.num_envs
        for worker_id in range(self.num_envs):
            t = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=f"env-worker-{worker_id}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()
        logger.info("Started %d env workers", self.num_envs)

    def stop(self, join_timeout: float = 10.0) -> None:
        """Signal workers to stop and join them.

        env.step() has no client-side gRPC deadline, so a straggler is
        logged rather than raised; threads are daemons so the process can
        always exit.
        """
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=join_timeout)
            if t.is_alive():
                logger.warning("Worker %s did not stop within %.1fs", t.name, join_timeout)
        self._threads = []

    @property
    def total_env_steps(self) -> int:
        return self.replay_buffer.total_pushed

    @property
    def total_episodes(self) -> int:
        with self._stats_lock:
            return self._total_episodes

    @property
    def alive_workers(self) -> int:
        with self._stats_lock:
            return self._alive_workers

    def pop_episode_results(self) -> List[Tuple[float, int, int, str]]:
        """Drain and return finished-episode results since the last call."""
        with self._stats_lock:
            results = self._episode_results
            self._episode_results = []
            return results

    def _create_env(self, worker_id: int, old_env: Any = None) -> Any:
        """(Re)create one worker's env with retries (pattern from
        train_dqn_robust.reset_environment)."""
        for attempt in range(self.max_env_retries):
            try:
                if old_env is not None:
                    try:
                        old_env.close()
                    except Exception:
                        pass
                    old_env = None
                env = self.env_factory(worker_id)
                logger.info("Worker %d: environment (re)created", worker_id)
                return env
            except Exception as e:
                logger.warning(
                    "Worker %d: env creation attempt %d/%d failed: %s",
                    worker_id, attempt + 1, self.max_env_retries, e,
                )
                if attempt < self.max_env_retries - 1:
                    time.sleep(2)
        raise RuntimeError(f"Worker {worker_id}: failed to create environment")

    def _worker_loop(self, worker_id: int) -> None:
        # Private RNG per worker: module-level random/np.random shared
        # across threads would correlate exploration between workers.
        rng = random.Random(self.seed * 1000 + worker_id)
        env = None
        try:
            env = self._create_env(worker_id)
            while not self._stop_event.is_set():
                try:
                    self._run_episode(worker_id, env, rng)
                except Exception as e:
                    if self._stop_event.is_set():
                        break
                    logger.warning("Worker %d: episode failed: %s", worker_id, e)
                    env = self._create_env(worker_id, old_env=env)
        except Exception as e:
            logger.error("Worker %d: dying: %s", worker_id, e)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            with self._stats_lock:
                self._alive_workers -= 1
            logger.info("Worker %d: exited", worker_id)

    def _run_episode(self, worker_id: int, env: Any, rng: random.Random) -> None:
        n_actions = env.action_space.n
        state, info = env.reset()
        valid_mask = info.get("valid_actions_mask", np.ones(n_actions, dtype=bool))

        episode_reward = 0.0
        episode_length = 0
        outcome = "draw"  # episodes that hit a step/turn cap count as draws

        while episode_length < self.max_steps_per_episode:
            if self._stop_event.is_set():
                return  # partial episode: pushed transitions are still valid
            action = self.action_fn(state, valid_mask, worker_id, rng)
            next_state, reward, terminated, truncated, next_info = env.step(action)

            done = terminated or truncated
            valid_mask = next_info.get("valid_actions_mask", np.ones(n_actions, dtype=bool))

            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                if terminated:
                    winner = next_info.get("winner")
                    outcome = "win" if winner == getattr(env, "player_id", 0) else "loss"
                break

        with self._stats_lock:
            self._total_episodes += 1
            self._episode_results.append((episode_reward, episode_length, worker_id, outcome))
