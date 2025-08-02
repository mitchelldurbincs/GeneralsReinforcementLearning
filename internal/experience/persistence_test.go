package experience

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/timestamppb"
)

func TestFilePersistence_Creation(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultPersistenceConfig()
	config.Type = PersistenceTypeFile
	config.BaseDir = tempDir
	
	fp, err := NewFilePersistence(config, logger)
	require.NoError(t, err)
	defer fp.Close()
	
	assert.NotNil(t, fp)
	assert.Equal(t, config, fp.config)
	
	// Check that base directory was created
	_, err = os.Stat(tempDir)
	assert.NoError(t, err)
}

func TestFilePersistence_WriteAndRead(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultPersistenceConfig()
	config.Type = PersistenceTypeFile
	config.BaseDir = tempDir
	
	fp, err := NewFilePersistence(config, logger)
	require.NoError(t, err)
	defer fp.Close()
	
	// Create test experiences
	gameID := "test-game-123"
	experiences := []*experiencepb.Experience{
		{
			ExperienceId: "exp-1",
			GameId:       gameID,
			PlayerId:     1,
			Turn:         10,
			State: &experiencepb.TensorState{
				Shape: []int32{9, 10, 10},
				Data:  make([]float32, 900),
			},
			Action: 5,
			Reward: 0.5,
			NextState: &experiencepb.TensorState{
				Shape: []int32{9, 10, 10},
				Data:  make([]float32, 900),
			},
			Done:        false,
			ActionMask:  make([]bool, 400),
			CollectedAt: timestamppb.Now(),
		},
		{
			ExperienceId: "exp-2",
			GameId:       gameID,
			PlayerId:     1,
			Turn:         11,
			State: &experiencepb.TensorState{
				Shape: []int32{9, 10, 10},
				Data:  make([]float32, 900),
			},
			Action: 15,
			Reward: -0.2,
			NextState: &experiencepb.TensorState{
				Shape: []int32{9, 10, 10},
				Data:  make([]float32, 900),
			},
			Done:        true,
			ActionMask:  make([]bool, 400),
			CollectedAt: timestamppb.Now(),
		},
	}
	
	// Write experiences
	ctx := context.Background()
	err = fp.Write(ctx, experiences)
	require.NoError(t, err)
	
	// Verify stats
	stats := fp.Stats()
	assert.Equal(t, int64(2), stats.TotalWritten)
	assert.Greater(t, stats.BytesWritten, int64(0))
	
	// Read experiences back
	readExps, err := fp.Read(ctx, gameID, 10)
	require.NoError(t, err)
	assert.Len(t, readExps, 2)
	
	// Verify content
	assert.Equal(t, "exp-1", readExps[0].ExperienceId)
	assert.Equal(t, "exp-2", readExps[1].ExperienceId)
	assert.Equal(t, gameID, readExps[0].GameId)
	assert.Equal(t, gameID, readExps[1].GameId)
}

func TestFilePersistence_FileRotation(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultPersistenceConfig()
	config.Type = PersistenceTypeFile
	config.BaseDir = tempDir
	config.MaxFileSize = 1024 // Small size to trigger rotation
	
	fp, err := NewFilePersistence(config, logger)
	require.NoError(t, err)
	defer fp.Close()
	
	// Create a large experience to trigger rotation
	largeData := make([]float32, 1000)
	experiences := []*experiencepb.Experience{
		{
			ExperienceId: "exp-large",
			GameId:       "test-game",
			State: &experiencepb.TensorState{
				Shape: []int32{1, 100, 100},
				Data:  largeData,
			},
			NextState: &experiencepb.TensorState{
				Shape: []int32{1, 100, 100},
				Data:  largeData,
			},
			CollectedAt: timestamppb.Now(),
		},
	}
	
	// Write multiple times to trigger rotation
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		err = fp.Write(ctx, experiences)
		require.NoError(t, err)
	}
	
	// Check that multiple files were created
	files, err := filepath.Glob(filepath.Join(tempDir, "experiences_*.json"))
	require.NoError(t, err)
	assert.Greater(t, len(files), 1, "Expected multiple files due to rotation")
}

func TestFilePersistence_Delete(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultPersistenceConfig()
	config.Type = PersistenceTypeFile
	config.BaseDir = tempDir
	
	fp, err := NewFilePersistence(config, logger)
	require.NoError(t, err)
	defer fp.Close()
	
	// Write experiences for multiple games
	ctx := context.Background()
	games := []string{"game-1", "game-2", "game-3"}
	
	for _, gameID := range games {
		exp := &experiencepb.Experience{
			ExperienceId: "exp-" + gameID,
			GameId:       gameID,
			CollectedAt:  timestamppb.Now(),
		}
		err = fp.Write(ctx, []*experiencepb.Experience{exp})
		require.NoError(t, err)
	}
	
	// Delete experiences for game-2 (currently no-op)
	err = fp.Delete(ctx, "game-2")
	require.NoError(t, err)
	
	// Since delete is not implemented, all experiences should still exist
	exps, err := fp.Read(ctx, "game-2", 10)
	require.NoError(t, err)
	assert.Len(t, exps, 1) // game-2 experience still exists
	
	// Verify other games also exist
	exps, err = fp.Read(ctx, "game-1", 10)
	require.NoError(t, err)
	assert.Len(t, exps, 1)
}

func TestFilePersistence_TimedRotation(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	tempDir := t.TempDir()
	
	config := DefaultPersistenceConfig()
	config.Type = PersistenceTypeFile
	config.BaseDir = tempDir
	config.RotationInterval = 100 * time.Millisecond // Short interval for testing
	
	fp, err := NewFilePersistence(config, logger)
	require.NoError(t, err)
	defer fp.Close()
	
	// Write initial experience
	ctx := context.Background()
	exp := &experiencepb.Experience{
		ExperienceId: "exp-1",
		GameId:       "test",
		CollectedAt:  timestamppb.Now(),
	}
	err = fp.Write(ctx, []*experiencepb.Experience{exp})
	require.NoError(t, err)
	
	// Wait for rotation
	time.Sleep(150 * time.Millisecond)
	
	// Write another experience
	exp2 := &experiencepb.Experience{
		ExperienceId: "exp-2",
		GameId:       "test",
		CollectedAt:  timestamppb.Now(),
	}
	err = fp.Write(ctx, []*experiencepb.Experience{exp2})
	require.NoError(t, err)
	
	// Check that multiple files exist
	files, err := filepath.Glob(filepath.Join(tempDir, "experiences_*.json"))
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(files), 2, "Expected multiple files due to timed rotation")
}

func TestPersistenceConfig_Default(t *testing.T) {
	config := DefaultPersistenceConfig()
	
	assert.Equal(t, PersistenceTypeNone, config.Type)
	assert.Equal(t, OverflowStrategyDropOldest, config.OverflowStrategy)
	assert.Equal(t, "experiences", config.BaseDir)
	assert.Equal(t, int64(100*1024*1024), config.MaxFileSize)
	assert.Equal(t, 1*time.Hour, config.RotationInterval)
	assert.False(t, config.CompressionEnabled)
	assert.Equal(t, 1000, config.BatchSize)
	assert.Equal(t, 30*time.Second, config.FlushInterval)
}

func TestNewPersistenceLayer(t *testing.T) {
	logger := zerolog.New(zerolog.NewTestWriter(t))
	
	tests := []struct {
		name    string
		config  PersistenceConfig
		wantErr bool
		wantType interface{}
	}{
		{
			name: "None type",
			config: PersistenceConfig{
				Type: PersistenceTypeNone,
			},
			wantType: &NullPersistence{},
		},
		{
			name: "File type",
			config: PersistenceConfig{
				Type:    PersistenceTypeFile,
				BaseDir: t.TempDir(),
			},
			wantType: &FilePersistence{},
		},
		{
			name: "S3 type (not implemented)",
			config: PersistenceConfig{
				Type: PersistenceTypeS3,
			},
			wantErr: true,
		},
		{
			name: "Invalid type",
			config: PersistenceConfig{
				Type: "invalid",
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := NewPersistenceLayer(tt.config, logger)
			
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			
			require.NoError(t, err)
			assert.IsType(t, tt.wantType, layer)
			
			if fp, ok := layer.(*FilePersistence); ok {
				defer fp.Close()
			}
		})
	}
}

func TestNullPersistence(t *testing.T) {
	np := &NullPersistence{}
	ctx := context.Background()
	
	// All operations should be no-ops
	err := np.Write(ctx, nil)
	assert.NoError(t, err)
	
	exps, err := np.Read(ctx, "any", 10)
	assert.NoError(t, err)
	assert.Nil(t, exps)
	
	err = np.Delete(ctx, "any")
	assert.NoError(t, err)
	
	err = np.Close()
	assert.NoError(t, err)
	
	stats := np.Stats()
	assert.Equal(t, PersistenceStats{}, stats)
}