# Multi-Agent Training Architecture

This document explains how the event-driven architecture and state machine will support multi-agent reinforcement learning with matchmaking.

## Overview

The system will support:
- Thousands of concurrent games
- Matchmaking based on skill/ELO ratings
- Self-play and league play
- Distributed training across multiple machines
- Real-time experience collection
- Dynamic agent pool management

## Architecture Components

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Agent Pool        │     │   Matchmaker        │     │   Game Manager      │
│                     │     │                     │     │                     │
│ - Agent 1 (ELO 1200)│◄────┤ - Skill matching    │────►│ - Game instances    │
│ - Agent 2 (ELO 1150)│     │ - Queue management  │     │ - Resource limits   │
│ - Agent 3 (ELO 1300)│     │ - Tournament mode   │     │ - Game lifecycle    │
│ - Human players     │     │ - Self-play pairing │     │                     │
└─────────────────────┘     └─────────────────────┘     └──────────┬──────────┘
                                                                     │
                                    ┌────────────────────────────────┴───────────┐
                                    │                                            │
                            ┌───────▼────────┐                          ┌───────▼────────┐
                            │   Game #1      │                          │   Game #N      │
                            │                │                          │                │
                            │ State Machine  │                          │ State Machine  │
                            │ Event Bus      │                          │ Event Bus      │
                            │ Engine         │                          │ Engine         │
                            └───────┬────────┘                          └───────┬────────┘
                                    │                                            │
                            ┌───────▼────────┐                          ┌───────▼────────┐
                            │ Experience     │                          │ Experience     │
                            │ Collector      │                          │ Collector      │
                            └───────┬────────┘                          └───────┬────────┘
                                    │                                            │
                                    └────────────────────┬───────────────────────┘
                                                         │
                                                ┌────────▼────────┐
                                                │ Replay Buffer   │
                                                │                 │
                                                │ - Prioritized   │
                                                │ - Distributed   │
                                                └─────────────────┘
```

## Event Flow for Multi-Agent Games

### 1. Matchmaking Phase

```go
// Matchmaker finds compatible agents
type MatchFoundEvent struct {
    MatchID   string
    Players   []PlayerInfo  // Includes agent IDs, ELO ratings
    GameMode  string       // "ranked", "self-play", "tournament"
}

// Game Manager creates new game instance
gameID := generateGameID()
game := NewGameInstance(gameID, matchEvent.Players)

// State: PhaseInitializing → PhaseLobby
game.StateMachine.TransitionTo(PhaseLobby, "Match created")
```

### 2. Agent Connection Phase

```go
// Agents join their assigned game
type AgentJoinedEvent struct {
    GameID   string
    AgentID  string
    Metadata map[string]interface{} // Model version, hyperparams, etc.
}

// When all agents connected
if game.AllAgentsReady() {
    // State: PhaseLobby → PhaseStarting
    game.StateMachine.TransitionTo(PhaseStarting, "All agents connected")
}
```

### 3. Game Running Phase

```go
// Parallel game processing
for _, game := range activeGames {
    go func(g *GameInstance) {
        // Each game runs independently
        g.ProcessTurn()
        
        // Events flow to experience collectors
        g.EventBus.Publish(TurnCompletedEvent{
            GameID:    g.ID,
            Turn:      g.Turn,
            States:    g.GetStatesForAllPlayers(),
            Actions:   g.LastActions,
            Rewards:   g.CalculateRewards(),
        })
    }(game)
}
```

### 4. Experience Collection

```go
// Experience collector subscribes to game events
type ExperienceCollector struct {
    gameID      string
    buffer      *ReplayBuffer
    agentStates map[int]*AgentExperience
}

func (ec *ExperienceCollector) OnEvent(event Event) {
    switch e := event.(type) {
    case *MoveExecutedEvent:
        // Record state-action pairs
        ec.recordMove(e)
        
    case *CombatResolvedEvent:
        // Update rewards based on combat outcomes
        ec.updateRewards(e)
        
    case *GameEndedEvent:
        // Calculate final rewards, add to replay buffer
        ec.finalizeEpisode(e)
    }
}
```

## State Machine in Multi-Agent Context

### Game Lifecycle Management

```go
type GameInstance struct {
    ID           string
    StateMachine *states.StateMachine
    Engine       *game.Engine
    EventBus     *events.EventBus
    Agents       map[int]AgentConnection
}

// Parallel game states
Running Games:  [Running, Running, Running, ...]
Starting Games: [Starting, Lobby, Starting, ...]
Ended Games:    [Ended, Ending, Ended, ...]
```

### State Transitions for Training

```go
// Auto-reset for continuous training
func (g *GameInstance) HandleGameEnd() {
    // State: PhaseEnding → PhaseEnded
    g.StateMachine.TransitionTo(PhaseEnded, "Game complete")
    
    // Collect final experiences
    g.CollectFinalExperiences()
    
    // For training, auto-reset
    if g.TrainingMode {
        // State: PhaseEnded → PhaseReset → PhaseInitializing
        g.StateMachine.TransitionTo(PhaseReset, "Training reset")
        g.StateMachine.TransitionTo(PhaseInitializing, "New episode")
        
        // Reuse same agents for next game
        g.StartNewEpisode()
    }
}
```

## Matchmaking Integration

### Skill-Based Matching

```go
type Matchmaker struct {
    queues      map[string]*MatchQueue  // By skill range
    waitingPool map[string]*Agent
    elo         *ELOSystem
}

func (m *Matchmaker) FindMatch(agent *Agent) {
    // Place in appropriate queue
    skillRange := m.getSkillRange(agent.ELO)
    queue := m.queues[skillRange]
    
    // Check for available opponents
    if opponents := queue.FindCompatible(agent); len(opponents) > 0 {
        match := m.CreateMatch(agent, opponents)
        m.publishMatchFound(match)
    }
}
```

### Self-Play Mode

```go
func (m *Matchmaker) CreateSelfPlayMatch(agent *Agent, numCopies int) {
    players := make([]PlayerInfo, numCopies)
    for i := 0; i < numCopies; i++ {
        players[i] = PlayerInfo{
            ID:      fmt.Sprintf("%s_copy_%d", agent.ID, i),
            AgentID: agent.ID,
            ELO:     agent.ELO,
        }
    }
    
    match := Match{
        ID:      generateMatchID(),
        Mode:    "self-play",
        Players: players,
    }
    
    m.publishMatchFound(match)
}
```

## Distributed Training Support

### Game Distribution

```go
type GameOrchestrator struct {
    nodes         []ComputeNode
    loadBalancer  *LoadBalancer
    matchmaker    *Matchmaker
}

func (go *GameOrchestrator) AssignGame(match Match) {
    // Find least loaded node
    node := go.loadBalancer.SelectNode()
    
    // Create game on that node
    gameRequest := GameCreationRequest{
        MatchID: match.ID,
        Players: match.Players,
        Config:  go.getGameConfig(),
    }
    
    node.CreateGame(gameRequest)
}
```

### Experience Aggregation

```go
// Central replay buffer receives from all games
type DistributedReplayBuffer struct {
    shards      []BufferShard
    aggregator  *ExperienceAggregator
}

func (drb *DistributedReplayBuffer) AddExperience(exp Experience) {
    // Shard by game ID for distribution
    shard := drb.getShard(exp.GameID)
    shard.Add(exp)
    
    // Aggregate for training
    if drb.ShouldSample() {
        batch := drb.SampleBatch()
        drb.SendToTrainer(batch)
    }
}
```

## Event Subscribers for RL

### 1. Metrics Collector
```go
type MetricsCollector struct{}

func (mc *MetricsCollector) OnEvent(event Event) {
    switch e := event.(type) {
    case *GameEndedEvent:
        metrics.RecordGameLength(e.TotalTurns)
        metrics.RecordWinner(e.Winner)
        metrics.RecordFinalScores(e.Scores)
    case *StateTransitionEvent:
        metrics.RecordStateTransition(e.From, e.To)
    }
}
```

### 2. Model Evaluator
```go
type ModelEvaluator struct {
    currentModel  string
    winRates      map[string]float64
}

func (me *ModelEvaluator) OnEvent(event Event) {
    if e, ok := event.(*GameEndedEvent); ok {
        // Track win rates for A/B testing
        me.updateWinRate(e.WinnerModel)
        
        // Trigger model promotion if threshold met
        if me.shouldPromoteModel() {
            me.promoteModel()
        }
    }
}
```

### 3. Training Logger
```go
type TrainingLogger struct {
    tensorboard *TensorBoardLogger
}

func (tl *TrainingLogger) OnEvent(event Event) {
    // Log everything for debugging
    tl.tensorboard.LogEvent(event)
    
    // Special handling for training-relevant events
    switch e := event.(type) {
    case *ActionProcessedEvent:
        tl.logActionDistribution(e)
    case *RewardCalculatedEvent:
        tl.logRewardCurve(e)
    }
}
```

## Benefits for RL Training

### 1. **Episode Management**
- State machine ensures clean episode boundaries
- Auto-reset for continuous training
- Pause/resume for checkpointing

### 2. **Parallel Training**
- Thousands of games running simultaneously
- Independent state machines prevent interference
- Event-driven experience collection scales linearly

### 3. **Matchmaking for Curriculum Learning**
- Start agents against easier opponents
- Gradually increase difficulty
- Track progress via ELO ratings

### 4. **Debugging and Analysis**
- State history shows game progression
- Events provide detailed game traces
- Can replay specific games from events

### 5. **Fault Tolerance**
- Error state allows game recovery
- Failed games don't crash the system
- Can restart from checkpoints

## Implementation Priorities

1. **Phase 1**: Single-machine multi-game support
   - Game instance manager
   - Basic matchmaking
   - Local experience collection

2. **Phase 2**: Distributed game execution
   - gRPC-based game distribution
   - Centralized replay buffer
   - Load balancing

3. **Phase 3**: Advanced matchmaking
   - ELO system
   - Tournament support
   - League play

4. **Phase 4**: Training optimizations
   - Prioritized experience replay
   - Curriculum learning
   - Population-based training

This architecture will scale from running 10 games locally to thousands of games across a cluster, all while maintaining clean separation of concerns through events and states.