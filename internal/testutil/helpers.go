package testutil

import (
	"math/rand"
	"testing"

	"github.com/rs/zerolog"
)

// NewTestRNG creates a deterministic random number generator for tests
func NewTestRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}

// NopLogger returns a no-op logger for tests
func NopLogger() zerolog.Logger {
	return zerolog.Nop()
}

// AssertPanic asserts that the given function panics
func AssertPanic(t *testing.T, f func(), msgAndArgs ...interface{}) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic but none occurred: %v", msgAndArgs)
		}
	}()
	f()
}
