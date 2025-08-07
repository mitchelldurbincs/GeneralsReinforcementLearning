package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInit(t *testing.T) {
	// Create a temporary config file
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.yaml")

	configContent := `
game:
  map:
    city_ratio: 25
    city_start_army: 50
  production:
    general: 2
    city: 2
server:
  grpc_server:
    port: 8080
ui:
  window:
    width: 1024
    height: 768
`

	err := os.WriteFile(configFile, []byte(configContent), 0644)
	require.NoError(t, err)

	// Reset global state
	cfg = nil
	v = nil

	// Initialize config
	err = Init(configFile)
	require.NoError(t, err)

	// Test loaded values
	c := Get()
	assert.Equal(t, 25, c.Game.Map.CityRatio)
	assert.Equal(t, 50, c.Game.Map.CityStartArmy)
	assert.Equal(t, 2, c.Game.Production.General)
	assert.Equal(t, 2, c.Game.Production.City)
	assert.Equal(t, 8080, c.Server.GRPCServer.Port)
	assert.Equal(t, 1024, c.UI.Window.Width)
	assert.Equal(t, 768, c.UI.Window.Height)
}

func TestInitWithDefaults(t *testing.T) {
	// Reset global state
	cfg = nil
	v = nil

	// Initialize with non-existent config (should use defaults)
	err := Init("/non/existent/path/config.yaml")
	require.NoError(t, err)

	// Should have empty config (no defaults in code, only in yaml)
	c := Get()
	assert.NotNil(t, c)
}

func TestEnvironmentVariables(t *testing.T) {
	// Reset global state
	cfg = nil
	v = nil

	// Set environment variables
	os.Setenv("GRL_GAME_MAP_CITY_RATIO", "30")
	os.Setenv("GRL_SERVER_GRPC_SERVER_PORT", "9090")
	defer os.Unsetenv("GRL_GAME_MAP_CITY_RATIO")
	defer os.Unsetenv("GRL_SERVER_GRPC_SERVER_PORT")

	// Initialize config
	err := Init("")
	require.NoError(t, err)

	// Environment variables should override
	c := Get()
	assert.Equal(t, 30, c.Game.Map.CityRatio)
	assert.Equal(t, 9090, c.Server.GRPCServer.Port)
}

func TestSet(t *testing.T) {
	// Reset global state
	cfg = nil
	v = nil

	// Initialize config
	err := Init("")
	require.NoError(t, err)

	// Set values
	Set("game.map.city_ratio", 35)
	Set("ui.window.width", 1280)

	// Check updated values
	c := Get()
	assert.Equal(t, 35, c.Game.Map.CityRatio)
	assert.Equal(t, 1280, c.UI.Window.Width)
}

func TestGetHelpers(t *testing.T) {
	// Reset global state
	cfg = nil
	v = nil

	// Initialize config
	err := Init("")
	require.NoError(t, err)

	// Set some values
	Set("test.string", "hello")
	Set("test.int", 42)
	Set("test.bool", true)
	Set("test.float", 3.14)

	// Test getters
	assert.Equal(t, "hello", GetString("test.string"))
	assert.Equal(t, 42, GetInt("test.int"))
	assert.Equal(t, true, GetBool("test.bool"))
	assert.Equal(t, 3.14, GetFloat64("test.float"))
}

func TestLoadEnvironmentConfig(t *testing.T) {
	// Create temporary config files
	tmpDir := t.TempDir()

	// Base config
	baseConfig := filepath.Join(tmpDir, "config.yaml")
	baseContent := `
game:
  map:
    city_ratio: 20
server:
  grpc_server:
    port: 50051
`
	err := os.WriteFile(baseConfig, []byte(baseContent), 0644)
	require.NoError(t, err)

	// Environment-specific config
	envConfig := filepath.Join(tmpDir, "config.prod.yaml")
	envContent := `
game:
  map:
    city_ratio: 30
server:
  grpc_server:
    port: 8080
    log_level: "error"
`
	err = os.WriteFile(envConfig, []byte(envContent), 0644)
	require.NoError(t, err)

	// Change to temp directory
	oldWd, _ := os.Getwd()
	_ = os.Chdir(tmpDir)
	defer func() { _ = os.Chdir(oldWd) }()

	// Reset global state
	cfg = nil
	v = nil

	// Initialize base config
	err = Init(baseConfig)
	require.NoError(t, err)

	// Load environment config
	err = LoadEnvironmentConfig("prod")
	require.NoError(t, err)

	// Check merged values
	c := Get()
	assert.Equal(t, 30, c.Game.Map.CityRatio)              // Overridden
	assert.Equal(t, 8080, c.Server.GRPCServer.Port)        // Overridden
	assert.Equal(t, "error", c.Server.GRPCServer.LogLevel) // New value
}
