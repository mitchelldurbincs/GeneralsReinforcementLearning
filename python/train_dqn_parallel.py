#!/usr/bin/env python3
"""
Parallel DQN training for Generals.io: N environments collect experience
from worker threads into a shared replay buffer while the main thread runs
the learner. See documentation/claude/parallel-experience-collection-plan.md.

Workers select actions with their own torch.no_grad() forward passes on the
shared q_network. Concurrent inference reads on an nn.Module are safe; the
learner's optimizer.step() can occasionally produce a "torn" mix of old/new
weights mid-forward, which only perturbs exploration slightly and is the
accepted trade-off in async DQN collectors (Ape-X lineage). If this ever
needs to be airtight, snapshot state_dict() into a separate actor network
every K learner steps.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generals_gym import GeneralsEnv, ParallelEnvPool, ReplayBuffer
from train_dqn_robust import SimpleDQN


def worker_epsilon(worker_id, num_workers, eps_base=0.4, alpha=7.0):
    """Fixed per-worker exploration rates, Ape-X style.

    Gives epsilon in [eps_base .. eps_base^(1+alpha)] (~0.4 .. ~0.007)
    across workers regardless of pool size: no decay schedule, no shared
    state, and trivially correct across resumes.
    """
    if num_workers == 1:
        return eps_base
    return eps_base ** (1 + alpha * worker_id / (num_workers - 1))


class ParallelDQNTrainer:
    """Single learner consuming from a pool of parallel collectors."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        torch.manual_seed(config['seed'])

        # Training state
        self.learner_steps = 0
        self.env_steps_at_resume = 0
        self.episodes_at_resume = 0
        self.best_reward = -float('inf')

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []  # "win" / "loss" / "draw" per episode
        self.losses = []

        # Probe env to get observation/action shapes, then close it
        probe_env = self._make_env(0)
        obs_shape = probe_env.observation_space.shape
        self.n_actions = probe_env.action_space.n
        probe_env.close()

        self.q_network = SimpleDQN(obs_shape, self.n_actions).to(self.device)
        self.target_network = SimpleDQN(obs_shape, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])

        self.buffer = ReplayBuffer(config['buffer_size'])

        self.worker_epsilons = [
            worker_epsilon(i, config['num_envs'], config['epsilon_base'])
            for i in range(config['num_envs'])
        ]

        self.pool = ParallelEnvPool(
            num_envs=config['num_envs'],
            env_factory=self._make_env,
            action_fn=self._action_fn,
            replay_buffer=self.buffer,
            max_steps_per_episode=config['max_steps_per_episode'],
            seed=config['seed'],
        )

        self.checkpoint_dir = config.get('checkpoint_dir', 'dqn_checkpoints_parallel')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _make_env(self, worker_id):
        return GeneralsEnv(
            server_address=self.config['server_address'],
            board_width=self.config['board_width'],
            board_height=self.config['board_height'],
            max_players=2,
            fog_of_war=self.config.get('fog_of_war', True),
            max_turns=self.config.get('max_turns', 200),
            collect_experiences=self.config.get('collect_experiences', False),
        )

    def _action_fn(self, state, valid_mask, worker_id, rng):
        """Per-worker epsilon-greedy with action masking (runs in worker threads)."""
        if rng.random() < self.worker_epsilons[worker_id]:
            valid_actions = np.where(valid_mask)[0]
            if len(valid_actions) > 0:
                return int(rng.choice(list(valid_actions)))
            return 0
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            q_values[~valid_mask] = -float('inf')
            return int(np.argmax(q_values))

    def train_step(self):
        """Perform one training step (same loss/clipping as train_dqn_robust)."""
        if len(self.buffer) < self.config['batch_size']:
            return None

        batch = self.buffer.sample(self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + self.config['gamma'] * next_q * (1 - dones)

        # Huber loss: unclipped MSE diverged in long runs (see train_dqn_robust)
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    @property
    def total_env_steps(self):
        return self.env_steps_at_resume + self.buffer.total_pushed

    @property
    def total_episodes(self):
        return self.episodes_at_resume + self.pool.total_episodes

    def save_checkpoint(self, filename=None):
        """Save training checkpoint (same keys as train_dqn_robust plus learner_steps)."""
        if filename is None:
            filename = f"checkpoint_ep{self.total_episodes}.pth"

        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'episode': self.total_episodes,
            'total_steps': self.total_env_steps,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.config['epsilon_base'],
            'best_reward': self.best_reward,
            'config': self.config,
            'learner_steps': self.learner_steps,
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

        metrics_file = os.path.join(self.checkpoint_dir, 'training_metrics.json')
        recent_outcomes = self.episode_outcomes[-100:]
        metrics = {
            'episode_rewards': self.episode_rewards[-100:],
            'episode_lengths': self.episode_lengths[-100:],
            'avg_reward': float(np.mean(self.episode_rewards[-20:])) if self.episode_rewards else 0,
            'avg_length': float(np.mean(self.episode_lengths[-20:])) if self.episode_lengths else 0,
            'outcome_totals': {
                'win': self.episode_outcomes.count('win'),
                'loss': self.episode_outcomes.count('loss'),
                'draw': self.episode_outcomes.count('draw'),
            },
            'win_rate_last_100': (recent_outcomes.count('win') / len(recent_outcomes)
                                  if recent_outcomes else 0),
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def load_checkpoint(self, filepath):
        """Load training checkpoint."""
        # weights_only=False: checkpoints embed the config dict and numpy
        # scalars; these files are produced by this script and trusted
        checkpoint = torch.load(filepath, weights_only=False)

        # Resume offsets so ratio pacing doesn't see a phantom deficit
        # (the buffer restarts empty; total_pushed restarts at 0)
        self.episodes_at_resume = checkpoint['episode']
        self.env_steps_at_resume = checkpoint['total_steps']
        self.learner_steps = checkpoint.get('learner_steps', 0)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.best_reward = checkpoint.get('best_reward', -float('inf'))

        self.logger.info(f"Checkpoint loaded from {filepath}")

    def train(self):
        """Main learner loop; collection runs in pool worker threads."""
        config = self.config
        self.logger.info(
            f"Starting parallel training: {config['num_envs']} envs, "
            f"{config['episodes']} episodes, train ratio {config['train_ratio']}"
        )
        self.logger.info(f"Device: {self.device}")
        self.logger.info(
            "Worker epsilons: " + ", ".join(f"{e:.3f}" for e in self.worker_epsilons)
        )

        start_time = time.time()
        learner_steps_at_start = self.learner_steps
        last_log = start_time
        last_log_env_steps = 0
        last_log_learner_steps = learner_steps_at_start
        last_log_episodes = 0
        last_periodic_checkpoint = 0

        self.pool.start()
        try:
            while (self.pool.total_episodes < config['episodes']
                   and self.pool.alive_workers > 0):
                # Pace the learner to train_ratio gradient steps per env step,
                # counted within this run only (a resumed run must not inherit
                # the previous run's deficit/surplus and hammer a fresh, nearly
                # empty buffer). Counter comparison self-corrects: train
                # continuously while behind, sleep while ahead.
                run_learner_steps = self.learner_steps - learner_steps_at_start
                if (run_learner_steps < self.buffer.total_pushed * config['train_ratio']
                        and len(self.buffer) >= config['batch_size']):
                    loss = self.train_step()
                    if loss is not None:
                        self.losses.append(loss)
                        self.learner_steps += 1
                        # Target sync keyed to learner steps (not env steps) to
                        # preserve gradient-steps-between-syncs regardless of
                        # collection speed
                        if self.learner_steps % config['target_update_freq'] == 0:
                            self.target_network.load_state_dict(self.q_network.state_dict())
                else:
                    time.sleep(0.005)

                now = time.time()
                if now - last_log >= config['log_interval']:
                    for reward, length, _worker_id, outcome in self.pool.pop_episode_results():
                        self.episode_rewards.append(reward)
                        self.episode_lengths.append(length)
                        self.episode_outcomes.append(outcome)

                    env_steps = self.total_env_steps
                    episodes = self.pool.total_episodes
                    dt = now - last_log
                    steps_per_s = (env_steps - self.env_steps_at_resume - last_log_env_steps) / dt
                    eps_per_min = (episodes - last_log_episodes) / dt * 60
                    learner_per_s = (self.learner_steps - last_log_learner_steps) / dt
                    achieved_ratio = learner_per_s / steps_per_s if steps_per_s > 0 else 0
                    avg_reward = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0
                    avg_length = np.mean(self.episode_lengths[-20:]) if self.episode_lengths else 0
                    avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                    recent = self.episode_outcomes[-100:]
                    win_rate = recent.count('win') / len(recent) if recent else 0
                    draw_rate = recent.count('draw') / len(recent) if recent else 0

                    self.logger.info(
                        f"Episodes: {episodes}/{config['episodes']} | "
                        f"Env steps: {env_steps} ({steps_per_s:.1f}/s) | "
                        f"Eps/min: {eps_per_min:.1f} | "
                        f"Learner: {self.learner_steps} ({learner_per_s:.1f}/s, ratio {achieved_ratio:.2f}) | "
                        f"Buffer: {len(self.buffer)} | "
                        f"Workers: {self.pool.alive_workers}/{config['num_envs']} | "
                        f"Reward: {avg_reward:.2f} | Length: {avg_length:.1f} | "
                        f"Win/Draw(100): {win_rate:.2f}/{draw_rate:.2f} | "
                        f"Loss: {avg_loss:.4f}"
                    )

                    if self.episode_rewards and avg_reward > self.best_reward:
                        self.best_reward = float(avg_reward)
                        self.save_checkpoint('best_model.pth')

                    if episodes - last_periodic_checkpoint >= 50:
                        self.save_checkpoint()
                        last_periodic_checkpoint = episodes

                    last_log = now
                    last_log_env_steps = env_steps - self.env_steps_at_resume
                    last_log_learner_steps = self.learner_steps
                    last_log_episodes = episodes

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_checkpoint('interrupted.pth')
        finally:
            self.pool.stop()
            # Fold in any episodes that finished after the last log
            for reward, length, _worker_id, outcome in self.pool.pop_episode_results():
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.episode_outcomes.append(outcome)

        self.save_checkpoint('final_model.pth')

        elapsed = time.time() - start_time
        env_steps = self.total_env_steps - self.env_steps_at_resume
        self.logger.info(f"Training completed in {elapsed:.1f} seconds")
        self.logger.info(
            f"Final stats - Episodes: {self.total_episodes}, "
            f"Env steps: {self.total_env_steps}, Learner steps: {self.learner_steps}, "
            f"Avg throughput: {env_steps / elapsed:.1f} steps/s, "
            f"{self.pool.total_episodes / elapsed * 60:.1f} eps/min"
        )


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Parallel DQN training for Generals.io')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to train')
    parser.add_argument('--board-size', type=int, default=10, help='Board width and height')
    parser.add_argument('--max-turns', type=int, default=200, help='Max turns per game')
    parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--server', type=str, default='localhost:50051', help='Game server address')
    parser.add_argument('--checkpoint-dir', type=str, default='dqn_checkpoints_parallel', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint file to resume from')
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--train-ratio', type=float, default=1.0,
                        help='Learner gradient steps per env step (0 = pure collection benchmark; '
                             'lower to 0.25-0.5 if learner GIL pressure throttles collection at high N)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (exploration only; maps are server-side)')
    parser.add_argument('--log-interval', type=float, default=10.0, help='Seconds between log lines')
    parser.add_argument('--buffer-size', type=int, default=50000, help='Replay buffer capacity')
    parser.add_argument('--collect-experiences', action='store_true',
                        help='Enable server-side experience collection (stress-tests the collector path)')
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        'server_address': args.server,
        'board_width': args.board_size,
        'board_height': args.board_size,
        'fog_of_war': True,
        'max_turns': args.max_turns,
        'max_steps_per_episode': args.max_steps,
        'episodes': args.episodes,
        'num_envs': args.num_envs,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'log_interval': args.log_interval,
        'collect_experiences': args.collect_experiences,

        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon_base': 0.4,
        'buffer_size': args.buffer_size,
        'batch_size': 32,
        'target_update_freq': 2000,

        'checkpoint_dir': args.checkpoint_dir,
    }

    print("=" * 60)
    print("Parallel DQN Training for Generals.io")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    trainer = ParallelDQNTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    print("\n✓ Training complete!")


if __name__ == "__main__":
    import grpc
    from generals_pb.game.v1 import game_pb2, game_pb2_grpc

    # Check server
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = game_pb2_grpc.GameServiceStub(channel)
        stub.CreateGame(game_pb2.CreateGameRequest(
            config=game_pb2.GameConfig(width=5, height=5, max_players=2)
        ))
        channel.close()

        print("✓ Game server is running\n")
        main()

    except grpc.RpcError as e:
        print(f"\n✗ Error: Game server is not running!")
        print(f"  Details: {e}")
        print("  Please start the server with: go run cmd/game_server/main.go")
        sys.exit(1)
