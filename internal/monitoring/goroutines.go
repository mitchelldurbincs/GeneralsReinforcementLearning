package monitoring

import (
	"runtime"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
)

// GoroutineMonitor tracks goroutine metrics
type GoroutineMonitor struct {
	mu              sync.RWMutex
	baseline        int
	current         int
	peak            int
	checkInterval   time.Duration
	alertThreshold  int
	lastAlert       time.Time
	alertCooldown   time.Duration
	stopChan        chan struct{}
	componentCounts map[string]int
}

// NewGoroutineMonitor creates a new goroutine monitor
func NewGoroutineMonitor() *GoroutineMonitor {
	baseline := runtime.NumGoroutine()
	return &GoroutineMonitor{
		baseline:        baseline,
		current:         baseline,
		peak:            baseline,
		checkInterval:   30 * time.Second,
		alertThreshold:  1000,
		alertCooldown:   5 * time.Minute,
		stopChan:        make(chan struct{}),
		componentCounts: make(map[string]int),
	}
}

// Start begins monitoring goroutines
func (gm *GoroutineMonitor) Start() {
	go gm.monitor()
	log.Info().
		Int("baseline", gm.baseline).
		Msg("Started goroutine monitoring")
}

// Stop stops the monitor
func (gm *GoroutineMonitor) Stop() {
	close(gm.stopChan)
}

// monitor is the main monitoring loop
func (gm *GoroutineMonitor) monitor() {
	defer func() {
		if r := recover(); r != nil {
			log.Error().
				Interface("panic", r).
				Msg("Goroutine monitor panicked - restarting")
			time.Sleep(5 * time.Second)
			go gm.monitor()
		}
	}()

	ticker := time.NewTicker(gm.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			gm.checkGoroutines()
		case <-gm.stopChan:
			return
		}
	}
}

// checkGoroutines checks current goroutine count and alerts if needed
func (gm *GoroutineMonitor) checkGoroutines() {
	current := runtime.NumGoroutine()

	gm.mu.Lock()
	gm.current = current
	if current > gm.peak {
		gm.peak = current
	}

	// Check for potential leak
	growth := current - gm.baseline
	growthRate := float64(growth) / float64(gm.baseline) * 100

	shouldAlert := current > gm.alertThreshold &&
		time.Since(gm.lastAlert) > gm.alertCooldown

	if shouldAlert {
		gm.lastAlert = time.Now()
	}
	gm.mu.Unlock()

	// Log metrics
	log.Debug().
		Int("current", current).
		Int("baseline", gm.baseline).
		Int("peak", gm.peak).
		Float64("growth_rate", growthRate).
		Msg("Goroutine metrics")

	// Alert on high goroutine count
	if shouldAlert {
		log.Warn().
			Int("current", current).
			Int("threshold", gm.alertThreshold).
			Float64("growth_rate", growthRate).
			Msg("High goroutine count detected - possible leak")
	}
}

// RegisterComponent registers a component's goroutine count
func (gm *GoroutineMonitor) RegisterComponent(name string, count int) {
	gm.mu.Lock()
	defer gm.mu.Unlock()
	gm.componentCounts[name] = count
}

// GetMetrics returns current goroutine metrics
func (gm *GoroutineMonitor) GetMetrics() GoroutineMetrics {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	return GoroutineMetrics{
		Current:         gm.current,
		Baseline:        gm.baseline,
		Peak:            gm.peak,
		Growth:          gm.current - gm.baseline,
		ComponentCounts: copyMap(gm.componentCounts),
	}
}

// GoroutineMetrics contains goroutine statistics
type GoroutineMetrics struct {
	Current         int            `json:"current"`
	Baseline        int            `json:"baseline"`
	Peak            int            `json:"peak"`
	Growth          int            `json:"growth"`
	ComponentCounts map[string]int `json:"component_counts"`
}

func copyMap(m map[string]int) map[string]int {
	result := make(map[string]int, len(m))
	for k, v := range m {
		result[k] = v
	}
	return result
}
