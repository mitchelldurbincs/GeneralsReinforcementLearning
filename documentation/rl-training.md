# Reinforcement Learning Training Guide

This document covers RL-specific details including reward functions, neural network architectures, and training optimizations.

## Reward Function Design

### Basic Reward Structure
```python
# Win/Loss Rewards
WIN_GAME = 100.0
LOSE_GAME = -100.0

# Intermediate Rewards
CAPTURE_ENEMY_GENERAL = 20.0
CAPTURE_ENEMY_CITY = 2.0
CAPTURE_ENEMY_TILE = 0.5
LOSE_CITY = -2.0
LOSE_TILE = -0.5

# Per-Step Rewards
ARMY_GAIN_MULTIPLIER = 0.01  # reward = army_gain * multiplier
TIME_PENALTY = -0.1  # Per turn to encourage faster games
```

### Advanced Reward Shaping
- Territory control: Reward based on percentage of map controlled
- Strategic positioning: Bonus for controlling key map areas
- Army efficiency: Reward for maintaining army-to-territory ratio
- Exploration bonus: Small reward for revealing new map areas

## Neural Network Architecture

### Observation Space Design
Multi-channel tensor representation (9 channels for 20x20 board):
```python
# Channel definitions
CHANNEL_OWN_ARMIES = 0      # Own army counts
CHANNEL_ENEMY_ARMIES = 1    # Enemy army counts
CHANNEL_OWN_TERRITORY = 2   # Binary: owned tiles
CHANNEL_ENEMY_TERRITORY = 3 # Binary: enemy tiles
CHANNEL_NEUTRAL_TERRITORY = 4 # Binary: neutral tiles
CHANNEL_CITIES = 5          # City locations
CHANNEL_MOUNTAINS = 6       # Mountain locations
CHANNEL_VISIBLE = 7         # Visible tiles
CHANNEL_FOG = 8            # Fog of war
```

### Network Architecture Options

#### CNN-Based (Recommended for spatial reasoning)
```python
class GeneralsCNN(nn.Module):
    def __init__(self, board_size=20, num_channels=9):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Global features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Value and policy heads
        self.value_head = nn.Linear(128, 1)
        self.policy_head = nn.Linear(128, board_size * board_size * 4)  # 4 directions
```

#### GRU/LSTM-Based (For sequential decision making)
```python
class GeneralsGRU(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.value_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, num_actions)
```

#### Transformer-Based (For complex strategic planning)
```python
class GeneralsTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.pos_encoding = PositionalEncoding2D(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
```

## Production-Scale Training Optimizations

### 1. Batch Game Management
```python
class GamePool:
    """Pre-allocate and reuse game instances"""
    def __init__(self, pool_size=1000):
        self.games = [create_game() for _ in range(pool_size)]
        self.available = queue.Queue()
        for game in self.games:
            self.available.put(game)
    
    def get_game(self):
        game = self.available.get()
        game.reset()
        return game
    
    def return_game(self, game):
        self.available.put(game)
```

### 2. Vectorized Environment
```python
class VectorizedGeneralsEnv:
    """Run multiple games in parallel"""
    def __init__(self, num_envs=100):
        self.envs = [GeneralsEnv() for _ in range(num_envs)]
    
    def step(self, actions):
        """Step all environments in parallel"""
        states, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            s, r, d, _ = env.step(action)
            states.append(s)
            rewards.append(r)
            dones.append(d)
        return np.array(states), np.array(rewards), np.array(dones)
```

### 3. Experience Buffer Optimizations
```python
class OptimizedReplayBuffer:
    """High-performance experience storage"""
    def __init__(self, capacity=1_000_000):
        # Pre-allocate numpy arrays
        self.states = np.zeros((capacity, 9, 20, 20), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 9, 20, 20), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Use circular buffer
        self.ptr = 0
        self.size = 0
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
```

### 4. Distributed Training Architecture
```python
# Actor processes (CPU)
class Actor:
    def __init__(self, game_server_addr, model_server_addr):
        self.game_client = GameClient(game_server_addr)
        self.model_client = ModelClient(model_server_addr)
        
    async def collect_experience(self):
        # Get latest model
        model = await self.model_client.get_latest_model()
        
        # Play games and collect experience
        while True:
            state = await self.game_client.reset()
            done = False
            
            while not done:
                action = model.predict(state)
                next_state, reward, done = await self.game_client.step(action)
                
                # Send experience to learner
                experience = (state, action, reward, next_state, done)
                await self.send_to_learner(experience)
                
                state = next_state

# Learner process (GPU)
class Learner:
    def __init__(self, experience_queue, model_server):
        self.queue = experience_queue
        self.model = GeneralsCNN()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    async def train(self):
        while True:
            # Sample batch from experience queue
            batch = await self.queue.sample_batch(256)
            
            # Train step
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Publish updated model
            if self.steps % 1000 == 0:
                await self.model_server.publish_model(self.model)
```

## Performance Targets and Benchmarks

### Training Speed Targets
- **Local (Single GPU)**: 50k-100k game steps/second
- **Distributed (8 GPUs)**: 500k-1M game steps/second
- **Action latency**: <1ms for inference
- **Experience collection**: 10k+ experiences/second per actor

### Memory Requirements
- **Replay Buffer**: ~30GB for 1M experiences
- **Model**: ~50MB for CNN architecture
- **Game States**: ~15KB per game state

### Scaling Benchmarks
| Setup | Games/sec | Steps/sec | GPUs | CPUs |
|-------|-----------|-----------|------|------|
| Local | 100 | 50k | 1 | 8 |
| Small Cluster | 1000 | 500k | 4 | 64 |
| Large Cluster | 10000 | 5M | 32 | 512 |

## Training Pipeline

### 1. Curriculum Learning
Start with easier scenarios and gradually increase difficulty:
```python
curriculum = [
    {"map_size": 10, "fog_of_war": False, "num_opponents": 1},
    {"map_size": 15, "fog_of_war": False, "num_opponents": 1},
    {"map_size": 15, "fog_of_war": True, "num_opponents": 1},
    {"map_size": 20, "fog_of_war": True, "num_opponents": 1},
    {"map_size": 20, "fog_of_war": True, "num_opponents": 3},
]
```

### 2. Self-Play Training
```python
class SelfPlayTrainer:
    def __init__(self):
        self.current_model = GeneralsCNN()
        self.opponent_pool = []  # Historical versions
        
    def train_iteration(self):
        # Play against recent versions
        opponents = random.sample(self.opponent_pool[-10:], k=3)
        
        # Collect experience
        for opponent in opponents:
            experience = self.play_games(self.current_model, opponent)
            self.replay_buffer.add_batch(experience)
        
        # Train on collected experience
        self.train_on_buffer()
        
        # Add to opponent pool periodically
        if self.iteration % 100 == 0:
            self.opponent_pool.append(copy.deepcopy(self.current_model))
```

### 3. Evaluation Metrics
- **ELO Rating**: Track model strength over time
- **Win Rate**: Against fixed baseline agents
- **Game Length**: Average turns to victory
- **Territory Control**: Average map control percentage
- **Army Efficiency**: Army-to-territory ratio

## Hyperparameter Recommendations

### PPO Algorithm
```python
ppo_config = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}
```

### DQN Algorithm
```python
dqn_config = {
    "learning_rate": 1e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 50_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 10_000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}
```

## Monitoring and Debugging

### Key Metrics to Track
```python
metrics = {
    # Training metrics
    "loss/policy": policy_loss,
    "loss/value": value_loss,
    "loss/entropy": entropy_loss,
    "gradients/norm": grad_norm,
    
    # Game metrics
    "game/win_rate": wins / total_games,
    "game/avg_length": avg_game_length,
    "game/avg_reward": avg_episode_reward,
    
    # Performance metrics
    "perf/fps": frames_per_second,
    "perf/gpu_utilization": gpu_usage,
    "perf/memory_usage": memory_mb,
}
```

### TensorBoard Integration
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/generals_training')

for step in range(training_steps):
    # Training step...
    
    # Log metrics
    writer.add_scalar('Loss/Policy', policy_loss, step)
    writer.add_scalar('Loss/Value', value_loss, step)
    writer.add_scalar('Game/WinRate', win_rate, step)
    
    # Log game visualization
    if step % 1000 == 0:
        game_image = render_game_state(state)
        writer.add_image('Game/State', game_image, step)
```

## Common Issues and Solutions

### 1. Slow Training Speed
- Enable GPU acceleration for neural networks
- Use vectorized environments
- Optimize experience collection with batching
- Profile code to identify bottlenecks

### 2. Poor Sample Efficiency
- Implement prioritized experience replay
- Use n-step returns
- Add auxiliary tasks (e.g., state prediction)
- Tune hyperparameters carefully

### 3. Training Instability
- Clip gradients to prevent explosions
- Use normalized observations
- Implement proper exploration schedules
- Monitor value function estimates

### 4. Memory Issues
- Use memory-mapped replay buffers
- Implement experience compression
- Clean up old game instances
- Monitor memory leaks with profilers