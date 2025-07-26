# Implementation Roadmap for Multi-Agent RL System

## Current State
- ✅ Event system implemented
- ✅ State machine implemented  
- ⏳ Coordinate system (Phase 4 pending)
- 🔲 Multi-agent infrastructure

## Next Implementation Steps

### Step 1: Game Instance Manager (1 week)
Create infrastructure to run multiple games simultaneously on a single machine.

```go
// internal/game/manager/instance_manager.go
type InstanceManager struct {
    instances    map[string]*GameInstance
    maxGames     int
    eventBus     *events.EventBus  // Global event aggregator
}

type GameInstance struct {
    ID           string
    Engine       *game.Engine
    StateMachine *states.StateMachine
    EventBus     *events.EventBus  // Per-game events
    Players      []PlayerConnection
}
```

**Benefits**: Run 100+ games in parallel on a single machine for training.

### Step 2: Agent Interface (3-4 days)
Define how agents interact with games.

```go
// internal/agents/interface.go
type Agent interface {
    GetAction(state GameState) Action
    OnGameStart(gameID string, playerID int)
    OnGameEnd(result GameResult)
}

type RemoteAgent struct {
    ID       string
    Endpoint string  // gRPC endpoint
    Client   AgentClient
}
```

**Benefits**: Agents can be local (for testing) or remote (for distributed training).

### Step 3: Basic Matchmaker (3-4 days)
Simple matchmaking to create games.

```go
// internal/matchmaking/matchmaker.go
type Matchmaker struct {
    waitingQueue []AgentRequest
    gameManager  *InstanceManager
}

func (m *Matchmaker) TryMatch() {
    if len(m.waitingQueue) >= 2 {
        // Create 2-player game
        players := m.waitingQueue[:2]
        m.gameManager.CreateGame(players)
        m.waitingQueue = m.waitingQueue[2:]
    }
}
```

**Benefits**: Automated game creation for continuous training.

### Step 4: Experience Collector (1 week)
Collect training data from games.

```go
// internal/rl/experience/collector.go
type ExperienceCollector struct {
    gameID   string
    buffer   []Experience
    eventBus *events.EventBus
}

type Experience struct {
    State      []float32  // Tensor representation
    Action     int
    Reward     float32
    NextState  []float32
    Done       bool
    PlayerView int  // Which player's perspective
}
```

**Benefits**: Structured data collection for neural network training.

### Step 5: Replay Buffer (3-4 days)
Efficient storage and sampling of experiences.

```go
// internal/rl/replay/buffer.go
type ReplayBuffer struct {
    capacity   int
    buffer     []Experience
    priorities []float32  // For prioritized replay
}

func (rb *ReplayBuffer) Sample(batchSize int) []Experience {
    // Prioritized sampling based on TD error
}
```

**Benefits**: Improved sample efficiency for training.

## Integration Example

Here's how it all connects:

```go
func main() {
    // Create core components
    manager := manager.NewInstanceManager(maxGames=100)
    matchmaker := matchmaking.NewMatchmaker(manager)
    
    // Create training agents
    for i := 0; i < 200; i++ {
        agent := agents.NewNeuralAgent(modelPath)
        matchmaker.AddAgent(agent)
    }
    
    // Subscribe to collect experiences
    globalCollector := experience.NewGlobalCollector()
    manager.EventBus.Subscribe(globalCollector.CollectExperience)
    
    // Run training loop
    for {
        // Matchmaker creates games
        matchmaker.ProcessQueue()
        
        // Games run in parallel
        manager.UpdateAllGames()
        
        // Experiences flow to replay buffer
        if globalCollector.BufferSize() > minBatchSize {
            batch := globalCollector.SampleBatch()
            trainer.Train(batch)
        }
    }
}
```

## Why This Order?

1. **Instance Manager First**: Need to run multiple games before anything else matters
2. **Agent Interface**: Defines how games communicate with AI/human players  
3. **Matchmaker**: Automates game creation for training
4. **Experience Collection**: Captures data for learning
5. **Replay Buffer**: Optimizes training efficiency

Each component builds on the previous ones and can be tested independently.

## Immediate Next Action

Start with the Game Instance Manager since it's the foundation for everything else. Would you like me to implement it?