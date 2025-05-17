package game

import (
	"fmt"
	"math/rand"
	"strings"
    "time"
)

type Engine struct {
	gs  *GameState
	rng *rand.Rand
}

// ---------------------------------------------------------------------------
// ctor
// ---------------------------------------------------------------------------

func NewEngine(w, h, players int, rng *rand.Rand) *Engine {
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	e := &Engine{
		gs: &GameState{
			Board:   NewBoard(w, h),
			Players: make([]Player, players),
		},
		rng: rng,
	}

	e.placeCities()
	e.placeGenerals()

	return e
}

// ---------------------------------------------------------------------------

func (e *Engine) placeCities() {
	b := e.gs.Board
	want := (b.W * b.H) / CityRatio
	placed := 0
	for placed < want {
		x, y := e.rng.Intn(b.W), e.rng.Intn(b.H)
		idx := b.Idx(x, y)
		t := &b.T[idx]
		if t.IsNeutral() && t.Type == TileNormal {
			t.Type = TileCity
			t.Army = CityStartArmy
			placed++
		}
	}
}

func (e *Engine) placeGenerals() {
	b := e.gs.Board
	for pid := range e.gs.Players {
		e.gs.Players[pid] = Player{ID: pid, Alive: true, GeneralIdx: -1}

	TRY:
		for {
			x, y := e.rng.Intn(b.W), e.rng.Intn(b.H)
			idx := b.Idx(x, y)
			if !b.T[idx].IsNeutral() || b.T[idx].IsCity() {
				continue
			}
			// keep at least MinGeneralSpacing from other generals
			for _, p := range e.gs.Players {
				if p.GeneralIdx == -1 {
					continue
				}
				gx, gy := b.XY(p.GeneralIdx)
				if b.Distance(x, y, gx, gy) < MinGeneralSpacing {
					continue TRY
				}
			}
			t := &b.T[idx]
			t.Owner, t.Army, t.Type = pid, 1, TileGeneral
			e.gs.Players[pid].GeneralIdx = idx
			break
		}
	}
}

// ---------------------------------------------------------------------------
// one game tick
// ---------------------------------------------------------------------------

func (e *Engine) Step() {
	e.gs.Turn++
	e.processTurnProduction()
	e.updatePlayerStats()
}

// ---------------------------------------------------------------------------

func (e *Engine) processTurnProduction() {
	growNormal := e.gs.Turn%NormalGrowInterval == 0
	for i := range e.gs.Board.T {
		t := &e.gs.Board.T[i]
		if t.IsNeutral() {
			continue
		}
		switch t.Type {
		case TileGeneral:
			t.Army += GeneralProduction
		case TileCity:
			t.Army += CityProduction
		case TileNormal:
			if growNormal {
				t.Army += NormalProduction
			}
		}
	}
}

func (e *Engine) updatePlayerStats() {
	// reset
	for pid := range e.gs.Players {
		e.gs.Players[pid].ArmyCount = 0
	}
	for i, t := range e.gs.Board.T {
		if t.IsNeutral() {
			continue
		}
		p := &e.gs.Players[t.Owner]
		p.ArmyCount += t.Army
		if t.IsGeneral() {
			p.GeneralIdx = i
		}
	}

	// mark dead/alive
	for pid := range e.gs.Players {
		e.gs.Players[pid].Alive = e.gs.Players[pid].GeneralIdx != -1
	}
}

// ---------------------------------------------------------------------------
// public helpers
// ---------------------------------------------------------------------------

func (e *Engine) GameState() GameState { // shallow copy
	return *e.gs
}

func (e *Engine) Board() string {
	var sb strings.Builder
	for y := 0; y < e.gs.Board.H; y++ {
		for x := 0; x < e.gs.Board.W; x++ {
			t := e.gs.Board.T[e.gs.Board.Idx(x, y)]
			switch {
			case t.IsCity() && t.IsNeutral():
				sb.WriteString(fmt.Sprintf(" C%-2d", t.Army))
			case t.IsNeutral():
				sb.WriteString(" .  ")
			case t.IsGeneral():
				sb.WriteString(fmt.Sprintf(" %d:%dG", t.Owner, t.Army))
			case t.IsCity():
				sb.WriteString(fmt.Sprintf(" %d:%dC", t.Owner, t.Army))
			default:
				sb.WriteString(fmt.Sprintf(" %d:%-2d", t.Owner, t.Army))
			}
		}
		sb.WriteByte('\n')
	}
	return sb.String()
}