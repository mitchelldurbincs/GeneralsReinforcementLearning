package gameserver

import (
	"fmt"
	"testing"
	"time"

	"github.com/mitchelldurbincs/GeneralsReinforcementLearning/internal/experience"
	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestExperienceAggregator(t *testing.T) {
	t.Skip("Skipping test - requires refactoring SimpleCollector to expose buffer for testing")
	// TODO: Add a test-only method to SimpleCollector or refactor to make testing easier
}

func TestExperienceAggregatorBatching(t *testing.T) {
	t.Skip("Skipping test - requires refactoring SimpleCollector to expose buffer for testing")
	// TODO: Add integration test with actual game engine
}

func TestExperienceAggregatorShutdown(t *testing.T) {
	t.Skip("Skipping test - requires refactoring SimpleCollector to expose buffer for testing")
	// TODO: Add integration test with actual game engine
}
