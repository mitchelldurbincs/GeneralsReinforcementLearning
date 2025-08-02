package experience

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	experiencepb "github.com/mitchelldurbincs/GeneralsReinforcementLearning/pkg/api/experience/v1"
	"github.com/rs/zerolog"
	"google.golang.org/protobuf/encoding/protojson"
)

var (
	// ErrPersistenceNotConfigured is returned when persistence operations are attempted without configuration
	ErrPersistenceNotConfigured = errors.New("persistence layer not configured")
	// ErrInvalidPersistenceType is returned when an unknown persistence type is specified
	ErrInvalidPersistenceType = errors.New("invalid persistence type")
)

// PersistenceType represents the type of persistence backend
type PersistenceType string

const (
	// PersistenceTypeNone disables persistence
	PersistenceTypeNone PersistenceType = "none"
	// PersistenceTypeFile enables file-based persistence
	PersistenceTypeFile PersistenceType = "file"
	// PersistenceTypeS3 enables S3-based persistence (future)
	PersistenceTypeS3 PersistenceType = "s3"
)

// OverflowStrategy defines how to handle buffer overflow
type OverflowStrategy string

const (
	// OverflowStrategyDropOldest drops the oldest experiences when buffer is full
	OverflowStrategyDropOldest OverflowStrategy = "drop_oldest"
	// OverflowStrategyDropNewest drops new experiences when buffer is full
	OverflowStrategyDropNewest OverflowStrategy = "drop_newest"
	// OverflowStrategyPersist persists experiences to storage when buffer is full
	OverflowStrategyPersist OverflowStrategy = "persist"
)

// PersistenceConfig contains configuration for the persistence layer
type PersistenceConfig struct {
	Type             PersistenceType
	OverflowStrategy OverflowStrategy
	
	// File-based config
	BaseDir          string
	MaxFileSize      int64 // Max size per file in bytes
	RotationInterval time.Duration
	CompressionEnabled bool
	
	// Batch config
	BatchSize        int
	FlushInterval    time.Duration
	
	// S3 config (future)
	S3Bucket         string
	S3Prefix         string
}

// DefaultPersistenceConfig returns a default persistence configuration
func DefaultPersistenceConfig() PersistenceConfig {
	return PersistenceConfig{
		Type:             PersistenceTypeNone,
		OverflowStrategy: OverflowStrategyDropOldest,
		BaseDir:          "experiences",
		MaxFileSize:      100 * 1024 * 1024, // 100MB
		RotationInterval: 1 * time.Hour,
		CompressionEnabled: false,
		BatchSize:        1000,
		FlushInterval:    30 * time.Second,
	}
}

// PersistenceLayer defines the interface for persisting experiences
type PersistenceLayer interface {
	// Write persists a batch of experiences
	Write(ctx context.Context, experiences []*experiencepb.Experience) error
	
	// Read retrieves experiences from storage
	Read(ctx context.Context, gameID string, limit int) ([]*experiencepb.Experience, error)
	
	// Delete removes persisted experiences
	Delete(ctx context.Context, gameID string) error
	
	// Close cleanly shuts down the persistence layer
	Close() error
	
	// Stats returns persistence statistics
	Stats() PersistenceStats
}

// PersistenceStats contains statistics about persistence operations
type PersistenceStats struct {
	TotalWritten    int64
	TotalRead       int64
	TotalDeleted    int64
	BytesWritten    int64
	BytesRead       int64
	WriteErrors     int64
	ReadErrors      int64
	LastWriteTime   time.Time
	LastReadTime    time.Time
}

// FilePersistence implements file-based persistence
type FilePersistence struct {
	config    PersistenceConfig
	logger    zerolog.Logger
	
	mu        sync.RWMutex
	stats     PersistenceStats
	
	currentFile *os.File
	currentSize int64
	fileIndex   int
	
	closeChan chan struct{}
	wg        sync.WaitGroup
}

// NewFilePersistence creates a new file-based persistence layer
func NewFilePersistence(config PersistenceConfig, logger zerolog.Logger) (*FilePersistence, error) {
	// Create base directory if it doesn't exist
	if err := os.MkdirAll(config.BaseDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}
	
	fp := &FilePersistence{
		config:    config,
		logger:    logger.With().Str("component", "file_persistence").Logger(),
		closeChan: make(chan struct{}),
	}
	
	// Open initial file
	if err := fp.rotateFile(); err != nil {
		return nil, err
	}
	
	// Start rotation timer if configured
	if config.RotationInterval > 0 {
		fp.wg.Add(1)
		go fp.rotationLoop()
	}
	
	return fp, nil
}

// Write persists a batch of experiences to file
func (fp *FilePersistence) Write(ctx context.Context, experiences []*experiencepb.Experience) error {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	
	if fp.currentFile == nil {
		return ErrPersistenceNotConfigured
	}
	
	for _, exp := range experiences {
		// Check if we need to rotate file
		if fp.config.MaxFileSize > 0 && fp.currentSize >= fp.config.MaxFileSize {
			if err := fp.rotateFile(); err != nil {
				fp.stats.WriteErrors++
				return fmt.Errorf("failed to rotate file: %w", err)
			}
		}
		
		// Marshal experience to JSON
		data, err := protojson.Marshal(exp)
		if err != nil {
			fp.stats.WriteErrors++
			return fmt.Errorf("failed to marshal experience: %w", err)
		}
		
		// Write to file with newline delimiter
		n, err := fp.currentFile.Write(append(data, '\n'))
		if err != nil {
			fp.stats.WriteErrors++
			return fmt.Errorf("failed to write experience: %w", err)
		}
		
		fp.currentSize += int64(n)
		fp.stats.TotalWritten++
		fp.stats.BytesWritten += int64(n)
	}
	
	// Sync to disk
	if err := fp.currentFile.Sync(); err != nil {
		fp.logger.Warn().Err(err).Msg("Failed to sync file")
	}
	
	fp.stats.LastWriteTime = time.Now()
	
	fp.logger.Debug().
		Int("batch_size", len(experiences)).
		Int64("file_size", fp.currentSize).
		Msg("Wrote experience batch to file")
	
	return nil
}

// Read retrieves experiences from storage
func (fp *FilePersistence) Read(ctx context.Context, gameID string, limit int) ([]*experiencepb.Experience, error) {
	fp.mu.RLock()
	defer fp.mu.RUnlock()
	
	// Look for all experience files
	pattern := filepath.Join(fp.config.BaseDir, "experiences_*.json")
	files, err := filepath.Glob(pattern)
	if err != nil {
		fp.stats.ReadErrors++
		return nil, fmt.Errorf("failed to list files: %w", err)
	}
	
	var experiences []*experiencepb.Experience
	for _, file := range files {
		if limit > 0 && len(experiences) >= limit {
			break
		}
		
		exps, err := fp.readFile(file, gameID, limit-len(experiences))
		if err != nil {
			fp.stats.ReadErrors++
			fp.logger.Warn().
				Err(err).
				Str("file", file).
				Msg("Failed to read experience file")
			continue
		}
		
		experiences = append(experiences, exps...)
	}
	
	fp.stats.LastReadTime = time.Now()
	fp.stats.TotalRead += int64(len(experiences))
	
	return experiences, nil
}

// readFile reads experiences from a single file
func (fp *FilePersistence) readFile(filename, gameID string, limit int) ([]*experiencepb.Experience, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	var experiences []*experiencepb.Experience
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		if limit > 0 && len(experiences) >= limit {
			break
		}
		
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		
		var exp experiencepb.Experience
		if err := protojson.Unmarshal(line, &exp); err != nil {
			return nil, fmt.Errorf("failed to unmarshal experience: %w", err)
		}
		
		// Filter by game ID if specified
		if gameID == "" || exp.GameId == gameID {
			experiences = append(experiences, &exp)
			fp.stats.BytesRead += int64(len(line))
		}
	}
	
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}
	
	return experiences, nil
}

// Delete removes persisted experiences
func (fp *FilePersistence) Delete(ctx context.Context, gameID string) error {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	
	// For now, delete is not supported since we don't have gameID in filenames
	// In a real implementation, we would either:
	// 1. Include gameID in filename
	// 2. Read each file and delete only matching experiences
	// 3. Keep a metadata index of which files contain which games
	return nil
}

// rotateFile closes the current file and opens a new one
func (fp *FilePersistence) rotateFile() error {
	// Close current file if open
	if fp.currentFile != nil {
		if err := fp.currentFile.Close(); err != nil {
			fp.logger.Warn().Err(err).Msg("Failed to close previous file")
		}
	}
	
	// Generate new filename
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join(fp.config.BaseDir, fmt.Sprintf("experiences_%s_%d.json", timestamp, fp.fileIndex))
	fp.fileIndex++
	
	// Check if file already exists and adjust index
	for {
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			break
		}
		fp.fileIndex++
		filename = filepath.Join(fp.config.BaseDir, fmt.Sprintf("experiences_%s_%d.json", timestamp, fp.fileIndex))
	}
	
	// Open new file
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	
	fp.currentFile = file
	fp.currentSize = 0
	
	fp.logger.Info().
		Str("filename", filename).
		Msg("Rotated to new experience file")
	
	return nil
}

// rotationLoop handles periodic file rotation
func (fp *FilePersistence) rotationLoop() {
	defer fp.wg.Done()
	
	ticker := time.NewTicker(fp.config.RotationInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			fp.mu.Lock()
			if err := fp.rotateFile(); err != nil {
				fp.logger.Error().Err(err).Msg("Failed to rotate file")
			}
			fp.mu.Unlock()
			
		case <-fp.closeChan:
			return
		}
	}
}

// Close cleanly shuts down the persistence layer
func (fp *FilePersistence) Close() error {
	close(fp.closeChan)
	fp.wg.Wait()
	
	fp.mu.Lock()
	defer fp.mu.Unlock()
	
	if fp.currentFile != nil {
		return fp.currentFile.Close()
	}
	
	return nil
}

// Stats returns persistence statistics
func (fp *FilePersistence) Stats() PersistenceStats {
	fp.mu.RLock()
	defer fp.mu.RUnlock()
	return fp.stats
}

// NullPersistence is a no-op persistence layer
type NullPersistence struct{}

func (n *NullPersistence) Write(ctx context.Context, experiences []*experiencepb.Experience) error {
	return nil
}

func (n *NullPersistence) Read(ctx context.Context, gameID string, limit int) ([]*experiencepb.Experience, error) {
	return nil, nil
}

func (n *NullPersistence) Delete(ctx context.Context, gameID string) error {
	return nil
}

func (n *NullPersistence) Close() error {
	return nil
}

func (n *NullPersistence) Stats() PersistenceStats {
	return PersistenceStats{}
}

// NewPersistenceLayer creates a persistence layer based on configuration
func NewPersistenceLayer(config PersistenceConfig, logger zerolog.Logger) (PersistenceLayer, error) {
	switch config.Type {
	case PersistenceTypeNone:
		return &NullPersistence{}, nil
	case PersistenceTypeFile:
		return NewFilePersistence(config, logger)
	case PersistenceTypeS3:
		return nil, errors.New("S3 persistence not yet implemented")
	default:
		return nil, ErrInvalidPersistenceType
	}
}