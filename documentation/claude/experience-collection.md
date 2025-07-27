# Experience Collection Architecture

## Overview

The experience collection system is responsible for gathering training data from games for reinforcement learning. It needs to handle thousands of concurrent games, efficiently store experiences, and provide batched access for training.

## Architecture Components

### 1. Experience Data Model

```python
@dataclass
class Experience:
    """Single transition for RL training"""
    state: np.ndarray          # Current game state as tensor
    action: int                # Action index taken
    reward: float              # Immediate reward
    next_state: np.ndarray     # Next game state as tensor
    done: bool                 # Episode terminated
    
    # Additional metadata
    game_id: str
    player_id: str
    turn: int
    timestamp: float
    
    # For prioritized replay
    td_error: Optional[float] = None
    priority: Optional[float] = None
    
    # For multi-agent
    opponent_action: Optional[int] = None
    global_state: Optional[np.ndarray] = None
```

### 2. Collection Methods

#### Option A: Event-Based Collection (Recommended)
Integrate with your existing event system:

```python
class ExperienceCollector(BaseGameEventHandler):
    """Collects experiences from game events"""
    
    def __init__(self, buffer_manager: BufferManager):
        self.buffer_manager = buffer_manager
        self.current_states = {}  # game_id -> player_id -> state
        
    def on_state_update(self, old_state, new_state):
        # Extract experiences for all players
        for player_id in self.get_active_players(new_state):
            if self.should_collect(player_id):
                experience = self.create_experience(
                    old_state, new_state, player_id
                )
                self.buffer_manager.add(experience)
    
    def on_action_submitted(self, game_id, player_id, action):
        # Store action for experience creation
        self.pending_actions[(game_id, player_id)] = action
```

#### Option B: Agent-Side Collection
Agents collect their own experiences:

```python
class RLAgent(BaseAgentNew):
    def __init__(self, experience_client: ExperienceClient):
        self.experience_client = experience_client
        self.last_state = None
        
    def select_action(self, game_state):
        # Convert to tensor
        state_tensor = self.state_to_tensor(game_state)
        
        # Get action from policy
        action = self.policy.select_action(state_tensor)
        
        # Store experience if we have previous state
        if self.last_state is not None:
            experience = Experience(
                state=self.last_state,
                action=self.last_action,
                reward=self.calculate_reward(game_state),
                next_state=state_tensor,
                done=game_state.status != IN_PROGRESS
            )
            self.experience_client.add(experience)
        
        self.last_state = state_tensor
        self.last_action = action
        return action
```

### 3. Storage Architecture

#### Local Storage (Development/Small Scale)
```python
class LocalExperienceBuffer:
    """In-memory circular buffer for experiences"""
    
    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
        
    def add(self, experience: Experience):
        with self.lock:
            self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        with self.lock:
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))
```

#### Distributed Storage (Production)
```python
class DistributedExperienceBuffer:
    """Redis-backed experience buffer for distributed training"""
    
    def __init__(self, redis_client, s3_client):
        self.redis = redis_client  # For recent experiences
        self.s3 = s3_client       # For long-term storage
        self.local_cache = []     # Batching before upload
        
    def add(self, experience: Experience):
        # Add to local cache
        self.local_cache.append(experience)
        
        # Batch upload to Redis
        if len(self.local_cache) >= BATCH_SIZE:
            self._flush_to_redis()
    
    def _flush_to_redis(self):
        # Serialize experiences
        data = serialize_experiences(self.local_cache)
        
        # Push to Redis list
        self.redis.lpush('experience_buffer', data)
        
        # Trim to max size
        self.redis.ltrim('experience_buffer', 0, MAX_BUFFER_SIZE)
        
        self.local_cache.clear()
```

### 4. Experience Service (gRPC)

Add to your proto definitions:

```protobuf
service ExperienceService {
    // Single experience submission
    rpc RecordExperience(RecordExperienceRequest) returns (RecordExperienceResponse);
    
    // Batch submission (more efficient)
    rpc RecordExperienceBatch(RecordExperienceBatchRequest) returns (RecordExperienceBatchResponse);
    
    // Get batch for training
    rpc GetExperienceBatch(GetExperienceBatchRequest) returns (GetExperienceBatchResponse);
    
    // Get buffer statistics
    rpc GetBufferStats(GetBufferStatsRequest) returns (GetBufferStatsResponse);
}

message RecordExperienceRequest {
    bytes state = 1;           // Serialized tensor
    int32 action = 2;
    float reward = 3;
    bytes next_state = 4;      // Serialized tensor
    bool done = 5;
    
    // Metadata
    string game_id = 6;
    string player_id = 7;
    int32 turn = 8;
}

message GetExperienceBatchRequest {
    int32 batch_size = 1;
    SamplingStrategy strategy = 2;
    
    enum SamplingStrategy {
        UNIFORM = 0;
        PRIORITIZED = 1;
        RECENT_FIRST = 2;
    }
}
```

### 5. Integration Points

#### With AgentRunner
```python
class AgentRunner:
    def __init__(self, agent, experience_collector=None):
        self.experience_collector = experience_collector
        
        if experience_collector:
            # Register collector with event system
            self.event_dispatcher.register(experience_collector)
```

#### With Game Server
The game server can also collect experiences directly:

```go
// internal/rl/experience/collector.go
type ExperienceCollector struct {
    client ExperienceServiceClient
    buffer []Experience
}

func (ec *ExperienceCollector) OnStateUpdate(gameID string, state *GameState) {
    // Convert state to tensor representation
    stateTensor := ec.stateToTensor(state)
    
    // Create experiences for all players
    for _, player := range state.Players {
        if player.LastAction != nil {
            exp := &Experience{
                State:     player.LastState,
                Action:    player.LastAction,
                Reward:    ec.calculateReward(player, state),
                NextState: stateTensor,
                Done:      state.Status != IN_PROGRESS,
            }
            ec.buffer = append(ec.buffer, exp)
        }
    }
    
    // Batch send to experience service
    if len(ec.buffer) >= BATCH_SIZE {
        ec.flush()
    }
}
```

### 6. Deployment Architecture

```yaml
# docker-compose.yml for development
version: '3.8'
services:
  game-server:
    image: generals-game-server
    ports:
      - "50051:50051"
  
  experience-service:
    image: generals-experience-service
    ports:
      - "50052:50052"
    environment:
      - REDIS_URL=redis://redis:6379
      - BUFFER_SIZE=1000000
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data
  
  trainer:
    image: generals-trainer
    environment:
      - EXPERIENCE_SERVICE=experience-service:50052
    depends_on:
      - experience-service
```

### 7. Advanced Features

#### Prioritized Experience Replay
```python
class PrioritizedBuffer(LocalExperienceBuffer):
    """Buffer with TD-error based prioritization"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        
    def add(self, experience: Experience, td_error: float):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        super().add(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        # Sample based on priorities
        probs = np.array(self.priorities) ** beta
        probs /= probs.sum()
        
        indices = np.random.choice(
            len(self.buffer), batch_size, p=probs
        )
        
        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** -beta
        
        return experiences, weights, indices
```

#### Experience Compression
```python
def compress_experience(exp: Experience) -> bytes:
    """Compress experience for efficient storage"""
    # Quantize state arrays (float32 -> uint8)
    state_q = quantize_state(exp.state)
    next_state_q = quantize_state(exp.next_state)
    
    # Pack into efficient format
    data = struct.pack(
        'HHfB?',  # action, reward, done
        exp.action,
        int(exp.reward * 1000),  # Fixed point
        exp.done
    )
    
    # Add compressed states
    data += lz4.compress(state_q.tobytes())
    data += lz4.compress(next_state_q.tobytes())
    
    return data
```

### 8. Monitoring and Debugging

```python
class ExperienceMonitor:
    """Monitor experience collection health"""
    
    def __init__(self):
        self.metrics = {
            'experiences_collected': Counter(),
            'buffer_size': Gauge(),
            'sample_latency': Histogram(),
            'games_contributing': Set(),
        }
    
    def log_stats(self):
        logger.info(f"""
        Experience Collection Stats:
        - Total collected: {self.metrics['experiences_collected']}
        - Current buffer size: {self.metrics['buffer_size']}
        - Active games: {len(self.metrics['games_contributing'])}
        - Avg sample latency: {self.metrics['sample_latency'].mean()}ms
        """)
```

## Implementation Recommendations

### Phase 1: Local Collection (Week 1)
1. Implement `ExperienceCollector` event handler
2. Create in-memory `LocalExperienceBuffer`
3. Integrate with `AgentRunner`
4. Add basic monitoring

### Phase 2: Service Implementation (Week 2)
1. Define protobuf messages
2. Implement `ExperienceService` gRPC server
3. Add Redis backing for buffer
4. Create Python client library

### Phase 3: Distributed Features (Week 3)
1. Add S3 archival for long-term storage
2. Implement prioritized replay
3. Add compression
4. Scale testing with 100+ concurrent games

### Phase 4: Production Hardening (Week 4)
1. Add comprehensive monitoring
2. Implement buffer overflow handling
3. Add experience validation
4. Performance optimization

## Key Design Decisions

1. **Event-based collection** leverages your existing architecture
2. **Hybrid storage** (Redis + S3) balances speed and capacity
3. **Batching everywhere** for efficiency at scale
4. **Player-centric experiences** to handle partial observability
5. **Flexible sampling strategies** for different training algorithms

This architecture will scale from local development to distributed training with thousands of concurrent games.